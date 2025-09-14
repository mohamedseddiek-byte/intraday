# app.py — Clean rewrite: robust data fetch + strict/tunable signals + backtest + Streamlit UI
# Save as: C:\Users\Acer\OneDrive\Desktop\app.py

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

# =========================
# Global config / helpers
# =========================
st.set_page_config(page_title="Intraday Recommender (Strict & Safe)", layout="wide")

MAX_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730}
FALLBACKS = {
    "1m": ["2m", "5m"],
    "2m": ["5m", "15m"],
    "5m": ["15m", "30m"],
    "15m": ["30m", "60m"],
    "30m": ["15m", "60m"],
    "60m": ["30m", "15m"],
}

def cap_days(interval: str, days: int) -> int:
    return min(int(days), MAX_DAYS.get(interval, 60))

# =========================
# Parameters
# =========================
@dataclass
class SignalParams:
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    atr_mult_stop: float = 1.5
    atr_mult_target: float = 2.0
    risk_pct: float = 1.0
    equity: float = 10000.0
    min_confluence: int = 3  # used in non-strict mode

@dataclass
class StrictParams:
    rsi_long_low: float = 55.0
    rsi_long_high: float = 65.0
    rsi_short_low: float = 35.0
    rsi_short_high: float = 45.0
    vol_mult: float = 1.2          # Volume >= vol_mult * VolSMA20
    rr_min: float = 1.8            # Minimum R:R
    vwap_tolerance: float = 0.0    # e.g., 0.0005 = 5 bps
    require_trend_stack: bool = True
    require_macd_zero_side: bool = True
    require_macd_rising: bool = True
    session_guard: bool = True     # if RTH: skip 09:30–10:00 and 15:30–16:00

# =========================
# Indicators
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    roll_up = up.rolling(period).mean()
    roll_dn = dn.rolling(period).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    f = ema(series, fast)
    s = ema(series, slow)
    line = f - s
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["Close"] * df["Volume"]).cumsum()
    vv = df["Volume"].cumsum().replace(0, np.nan)
    return pv / vv

# =========================
# Data loader (robust)
# =========================
def _download(ticker: str, interval: str, days: int) -> pd.DataFrame:
    days = cap_days(interval, days)
    df = yf.download(
        tickers=ticker,
        period=f"{days}d",
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)

    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")
    except Exception:
        pass

    if "Volume" in df.columns:
        df = df[df["Volume"] > 0]

    return df

def fetch_intraday(ticker: str, interval: str = "5m", days: int = 10, rth_only: bool = False) -> pd.DataFrame:
    """
    Robust fetch:
      - auto-cap days
      - flatten columns, numeric cast
      - filter RTH if requested
      - auto-recover if empty: drop RTH, shrink days, fallback intervals
    """
    base = _download(ticker, interval, days)

    def apply_rth(dfi: pd.DataFrame) -> pd.DataFrame:
        if dfi.empty:
            return dfi
        try:
            return dfi.between_time("09:30", "16:00")
        except Exception:
            return dfi

    dfr = apply_rth(base) if rth_only else base.copy()
    if not dfr.empty:
        return dfr

    # Recovery path
    # 1) Try without RTH if RTH filtered it out
    if rth_only and not base.empty:
        return base

    # 2) Shrink days staircase
    for d_try in [min(cap_days(interval, days), 30), 14, 7, 5, 3]:
        if d_try <= 0:
            continue
        tmp = _download(ticker, interval, d_try)
        tmp = apply_rth(tmp) if rth_only else tmp
        if not tmp.empty:
            return tmp

    # 3) Fallback intervals
    for iv in FALLBACKS.get(interval, []):
        tmp = _download(ticker, iv, cap_days(iv, days))
        tmp = apply_rth(tmp) if rth_only else tmp
        if not tmp.empty:
            return tmp

    raise ValueError(f"After cleaning and retries, no usable data for {ticker}.")

# =========================
# Indicators compute
# =========================
def compute_indicators(df: pd.DataFrame, p: SignalParams) -> pd.DataFrame:
    out = df.copy()
    out["EMA_fast"] = ema(out["Close"], p.ema_fast)
    out["EMA_slow"] = ema(out["Close"], p.ema_slow)
    out["RSI"] = rsi(out["Close"], p.rsi_period)
    macd_line, macd_sig, macd_hist = macd(out["Close"], p.macd_fast, p.macd_slow, p.macd_signal)
    out["MACD"] = macd_line
    out["MACD_signal"] = macd_sig
    out["MACD_hist"] = macd_hist
    out["ATR"] = atr(out, p.atr_period)
    out["VWAP"] = vwap(out)
    out["VolSMA20"] = out["Volume"].rolling(20).mean()
    return out

# =========================
# Signal engine
# =========================
def _pos_size_long(entry: float, stop: float, equity: float, risk_pct: float) -> int:
    risk_amt = equity * (risk_pct / 100.0)
    per_share = max(entry - stop, 1e-9)
    return max(int(risk_amt // per_share), 0)

def _pos_size_short(entry: float, stop: float, equity: float, risk_pct: float) -> int:
    risk_amt = equity * (risk_pct / 100.0)
    per_share = max(stop - entry, 1e-9)
    return max(int(risk_amt // per_share), 0)

def generate_latest_signal(df: pd.DataFrame, p: SignalParams, strict: bool = False, sp: Optional[StrictParams] = None) -> Optional[dict]:
    need = max(p.ema_slow, p.rsi_period, p.macd_slow, p.atr_period, 20) + 5
    if len(df) < need:
        return None
    if strict and sp is None:
        sp = StrictParams()

    row = df.iloc[-2]   # last completed bar
    prev = df.iloc[-3]

    # Scalars
    close = float(row["Close"])
    emaf  = float(row["EMA_fast"])
    emas  = float(row["EMA_slow"])
    rsi_v = float(row["RSI"])
    macd_h  = float(row["MACD_hist"])
    macd_hp = float(prev["MACD_hist"])
    vwap_v  = float(row["VWAP"])
    atr_v   = float(row["ATR"])
    vol     = float(row["Volume"])
    vol_sma = float(row.get("VolSMA20", np.nan))

    if strict:
        # Optional session guard (if df is tz-aware)
        if sp.session_guard and getattr(df.index, "tz", None) is not None:
            t = df.index[-2].tz_convert("America/New_York").time()
            if not (t >= pd.Timestamp("10:00").time() and t <= pd.Timestamp("15:30").time()):
                return None

        # Strict gates
        trend_long  = (close > emaf > emas) if sp.require_trend_stack else (emaf > emas)
        trend_short = (close < emaf < emas) if sp.require_trend_stack else (emaf < emas)

        rsi_long_ok  = sp.rsi_long_low  <= rsi_v <= sp.rsi_long_high
        rsi_short_ok = sp.rsi_short_low <= rsi_v <= sp.rsi_short_high

        macd_side_long  = (macd_h > 0.0) if sp.require_macd_zero_side else True
        macd_side_short = (macd_h < 0.0) if sp.require_macd_zero_side else True
        macd_slope_long  = (macd_h > macd_hp) if sp.require_macd_rising else True
        macd_slope_short = (macd_h < macd_hp) if sp.require_macd_rising else True

        tol = sp.vwap_tolerance
        vwap_long  = close >= vwap_v * (1 - tol)
        vwap_short = close <= vwap_v * (1 + tol)

        vol_ok = (not np.isnan(vol_sma)) and (vol >= sp.vol_mult * vol_sma)

        long_ok  = trend_long  and rsi_long_ok  and macd_side_long  and macd_slope_long  and vwap_long  and vol_ok and (atr_v > 0)
        short_ok = trend_short and rsi_short_ok and macd_side_short and macd_slope_short and vwap_short and vol_ok and (atr_v > 0)

        side = "LONG" if long_ok else ("SHORT" if short_ok else None)
        if side is None:
            return None

        entry = close
        if side == "LONG":
            stop = entry - p.atr_mult_stop * atr_v
            target = entry + p.atr_mult_target * atr_v
            rr = (target - entry) / (entry - stop + 1e-9)
            size = _pos_size_long(entry, stop, p.equity, p.risk_pct)
        else:
            stop = entry + p.atr_mult_stop * atr_v
            target = entry - p.atr_mult_target * atr_v
            rr = (entry - target) / (stop - entry + 1e-9)
            size = _pos_size_short(entry, stop, p.equity, p.risk_pct)

        if rr < sp.rr_min:
            return None

        return {
            "timestamp": df.index[-2],
            "side": side,
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "target": round(target, 4),
            "atr": round(atr_v, 4),
            "rsi": round(rsi_v, 2),
            "macd_hist": round(macd_h, 4),
            "vwap": round(vwap_v, 4),
            "rr_ratio": round(float(rr), 2),
            "position_size": int(size),
        }

    # Non-strict: confluence
    condL = int(emaf > emas) + int(50 <= rsi_v <= 70) + int((macd_h > 0) and (macd_hp <= 0)) + int(close > vwap_v)
    condS = int(emaf < emas) + int(30 <= rsi_v <= 50) + int((macd_h < 0) and (macd_hp >= 0)) + int(close < vwap_v)
    side = None
    if condL >= p.min_confluence and condL > condS:
        side = "LONG"
    elif condS >= p.min_confluence and condS > condL:
        side = "SHORT"
    if side is None:
        return None

    entry = close
    if side == "LONG":
        stop = entry - p.atr_mult_stop * atr_v
        target = entry + p.atr_mult_target * atr_v
        size = _pos_size_long(entry, stop, p.equity, p.risk_pct)
        rr = (target - entry) / (entry - stop + 1e-9)
    else:
        stop = entry + p.atr_mult_stop * atr_v
        target = entry - p.atr_mult_target * atr_v
        size = _pos_size_short(entry, stop, p.equity, p.risk_pct)
        rr = (entry - target) / (stop - entry + 1e-9)

    return {
        "timestamp": df.index[-2],
        "side": side,
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "target": round(target, 4),
        "atr": round(float(atr_v), 4),
        "rsi": round(float(rsi_v), 2),
        "macd_hist": round(float(macd_h), 4),
        "vwap": round(float(vwap_v), 4),
        "rr_ratio": round(float(rr), 2),
        "position_size": int(size),
    }

# =========================
# Batch runner (recs)
# =========================
def run(symbols: List[str], interval: str, days: int, equity: float, risk_pct: float,
        min_confluence: int = 3, rth_only: bool = False, save_csv: Optional[str] = None,
        strict: bool = False, strict_params: Optional[Dict] = None) -> pd.DataFrame:

    out = []
    P = SignalParams(equity=float(equity), risk_pct=float(risk_pct), min_confluence=int(min_confluence))
    SP = StrictParams(**strict_params) if (strict and strict_params) else (StrictParams() if strict else None)

    for tkr in symbols:
        try:
            raw = fetch_intraday(tkr, interval=interval, days=days, rth_only=rth_only)
            dfi = compute_indicators(raw, P).dropna().copy()
            sig = generate_latest_signal(dfi, P, strict=strict, sp=SP)
            if sig is None:
                out.append({"Ticker": tkr, "Time(NY)": np.nan, "Side": "NONE",
                            "Entry": np.nan, "Stop": np.nan, "Target": np.nan,
                            "ATR": np.nan, "RSI": np.nan, "MACD_hist": np.nan, "VWAP": np.nan,
                            "RR": np.nan, "PositionSize": 0})
            else:
                ts = sig["timestamp"]
                try:
                    ts = ts.tz_convert("America/New_York")
                except Exception:
                    pass
                out.append({"Ticker": tkr, "Time(NY)": ts, "Side": sig["side"],
                            "Entry": sig["entry"], "Stop": sig["stop"], "Target": sig["target"],
                            "ATR": sig["atr"], "RSI": sig["rsi"], "MACD_hist": sig["macd_hist"], "VWAP": sig["vwap"],
                            "RR": sig["rr_ratio"], "PositionSize": sig["position_size"]})
        except Exception as e:
            out.append({"Ticker": tkr, "Time(NY)": np.nan, "Side": f"ERROR: {e}",
                        "Entry": np.nan, "Stop": np.nan, "Target": np.nan,
                        "ATR": np.nan, "RSI": np.nan, "MACD_hist": np.nan, "VWAP": np.nan,
                        "RR": np.nan, "PositionSize": 0})

    df_out = pd.DataFrame(out)
    if save_csv:
        try:
            df_out.to_csv(save_csv, index=False)
        except Exception as e:
            st.warning(f"Could not save CSV: {e}")
    return df_out

# =========================
# Backtest (walk-forward)
# =========================
def walk_signals(df: pd.DataFrame, P: SignalParams, strict: bool, strict_params: Optional[Dict]) -> pd.DataFrame:
    dfi = compute_indicators(df, P).dropna().copy()
    trades = []
    SP = StrictParams(**strict_params) if (strict and strict_params) else (StrictParams() if strict else None)

    need = max(P.ema_slow, P.rsi_period, P.macd_slow, P.atr_period, 20) + 5
    if len(dfi) < need + 2:
        return pd.DataFrame(columns=["EntryTime","ExitTime","Side","Entry","Stop","Target","Exit","R","BarsHeld"])

    in_trade = False
    side = None
    entry = stop = target = None
    entry_i = None

    for i in range(need, len(dfi) - 1):
        window = dfi.iloc[: i + 1]
        if not in_trade:
            sig = generate_latest_signal(window, P, strict=strict, sp=SP)
            if sig and sig["timestamp"] == window.index[-2]:
                side = sig["side"]
                entry = float(sig["entry"])
                stop = float(sig["stop"])
                target = float(sig["target"])
                entry_i = i - 1
                in_trade = True
        else:
            bar = dfi.iloc[i]
            high = float(bar["High"])
            low = float(bar["Low"])

            exit_price = None
            r = None
            bars_held = i - entry_i

            if side == "LONG":
                if low <= stop:
                    exit_price = stop; r = -1.0
                elif high >= target:
                    exit_price = target; r = (target - entry) / (entry - stop + 1e-12)
            else:
                if high >= stop:
                    exit_price = stop; r = -1.0
                elif low <= target:
                    exit_price = target; r = (entry - target) / (stop - entry + 1e-12)

            if exit_price is not None:
                trades.append({
                    "EntryTime": dfi.index[entry_i], "ExitTime": dfi.index[i],
                    "Side": side, "Entry": round(entry,4), "Stop": round(stop,4),
                    "Target": round(target,4), "Exit": round(exit_price,4),
                    "R": round(float(r),3), "BarsHeld": int(bars_held)
                })
                in_trade = False
                side = None; entry = stop = target = None; entry_i = None

    return pd.DataFrame(trades)

def summarize(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series({"trades":0,"win_rate":0.0,"avg_R":0.0,"expectancy_R":0.0,"cum_R":0.0,"max_dd_R":0.0,"median_bars":0,"avg_bars":0.0})
    wins = trades["R"] > 0
    win_rate = float(wins.mean())
    avg_R = float(trades["R"].mean())
    cum_R = float(trades["R"].sum())
    eq = trades["R"].cumsum(); peak = eq.cummax(); dd = eq - peak
    max_dd_R = float(dd.min()) if not dd.empty else 0.0
    return pd.Series({
        "trades": int(len(trades)),
        "win_rate": round(win_rate,3),
        "avg_R": round(avg_R,3),
        "expectancy_R": round(avg_R,3),
        "cum_R": round(cum_R,2),
        "max_dd_R": round(max_dd_R,2),
        "median_bars": int(trades["BarsHeld"].median()),
        "avg_bars": round(float(trades["BarsHeld"].mean()),1),
    })

# =========================
# Streamlit UI
# =========================
st.title("📈 Intraday Recommender (Strict & Safe)")

with st.sidebar:
    st.header("Settings")
    symbols_str = st.text_input("Symbols (space-separated)", "SPY TSLL")
    interval = st.selectbox("Interval", ["1m","2m","5m","15m","30m","60m"], index=2)
    days = st.number_input("Days (auto-capped by interval)", 1, 1000, 30, 1)
    rth_only = st.checkbox("Regular Trading Hours only (09:30–16:00 NY)", True)

    st.divider()
    st.subheader("Strict filters")
    strict = st.checkbox("Enable strict A+ mode", True)
    colA, colB = st.columns(2)
    with colA:
        rsi_long_low  = st.number_input("RSI long low", 45.0, 80.0, 55.0, 0.5)
        rsi_short_low = st.number_input("RSI short low", 20.0, 55.0, 35.0, 0.5)
        vol_mult = st.slider("Volume × SMA20 ≥", 1.0, 2.0, 1.2, 0.05)
        require_trend_stack = st.checkbox("Require Close>EMA20>EMA50", True)
    with colB:
        rsi_long_high  = st.number_input("RSI long high", 50.0, 90.0, 65.0, 0.5)
        rsi_short_high = st.number_input("RSI short high",30.0, 70.0, 45.0, 0.5)
        rr_min = st.slider("Min RR", 1.2, 3.0, 1.8, 0.1)
        vwap_tol_bps = st.slider("VWAP tolerance (bps)", 0, 20, 0)
    vwap_tol = vwap_tol_bps / 10000.0
    require_macd_zero   = st.checkbox("Require MACD hist correct side of 0", True)
    require_macd_slope  = st.checkbox("Require MACD hist rising/falling", True)
    session_guard       = st.checkbox("Skip first/last 30 mins (if RTH)", True)

    st.divider()
    min_conf = st.slider("# agreeing checks (non-strict path)", 2, 5, 4)
    equity = st.number_input("Equity", 1000, 5_000_000, 25_000, 500)
    risk_pct = st.slider("Risk % per trade", 0.1, 5.0, 1.0, 0.1)
    save_csv = st.text_input("Save recommendations CSV (optional path)", "")

symbols = [s.strip().upper() for s in symbols_str.split() if s.strip()]
days_capped = cap_days(interval, days)
strict_params = dict(
    rsi_long_low=rsi_long_low, rsi_long_high=rsi_long_high,
    rsi_short_low=rsi_short_low, rsi_short_high=rsi_short_high,
    vol_mult=vol_mult, rr_min=rr_min, vwap_tolerance=vwap_tol,
    require_trend_stack=require_trend_stack,
    require_macd_zero_side=require_macd_zero,
    require_macd_rising=require_macd_slope,
    session_guard=session_guard,
)

left, right = st.columns([1,1])

with left:
    st.subheader("Run Recommendations")
    if st.button("Run now"):
        with st.spinner("Working…"):
            df_out = run(
                symbols=symbols, interval=interval, days=days_capped,
                equity=float(equity), risk_pct=float(risk_pct),
                min_confluence=int(min_conf), rth_only=bool(rth_only),
                save_csv=(save_csv if save_csv.strip() else None),
                strict=bool(strict), strict_params=(strict_params if strict else None),
            )
        st.success("Done.")
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            st.dataframe(df_out, use_container_width=True)
            if save_csv.strip():
                st.info(f"Saved CSV → {save_csv}")
        else:
            st.info("No rows returned. Try a different interval/days, uncheck RTH, or gently relax one strict slider (e.g., Volume 1.1× or RR ≥ 1.6).")

with right:
    st.subheader("Backtest (single symbol)")
    bt_symbol = st.selectbox("Symbol", options=(symbols if symbols else ["SPY"]), index=0)
    bt_days = st.number_input("Backtest days", 1, 1000, 60, 1)
    bt_days_capped = cap_days(interval, bt_days)
    save_trades = st.text_input("Save trades CSV (optional path)", "")

    if st.button("Run backtest"):
        try:
            with st.spinner("Backtesting…"):
                raw = fetch_intraday(bt_symbol, interval=interval, days=bt_days_capped, rth_only=bool(rth_only))
                P = SignalParams(equity=float(equity), risk_pct=float(risk_pct), min_confluence=int(min_conf))
                trades = walk_signals(raw, P, strict=bool(strict), strict_params=(strict_params if strict else None))
                summary = summarize(trades)
            st.success("Backtest complete.")
            st.write("**Summary**")
            st.dataframe(summary.to_frame("value"))
            if not trades.empty:
                st.write("**First 10 trades**")
                st.dataframe(trades.head(10), use_container_width=True)
                if save_trades.strip():
                    try:
                        trades.to_csv(save_trades, index=False)
                        st.info(f"Saved trades → {save_trades}")
                    except Exception as e:
                        st.warning(f"Could not save trades CSV: {e}")
            else:
                st.info("No trades met the strict criteria for the chosen window/interval.")
        except Exception as e:
            st.error(f"Backtest error: {e}")
