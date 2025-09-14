# app.py — One-file, cloud-stable Intraday Recommender + Scanner
# Fixes: pinned behavior, cache-bypass, retries, explicit TZ, S&P500 auto, presets, strict/non-strict
import time
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from typing import Optional, Tuple, List

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Intraday Recommender & Scanner", page_icon="📈", layout="wide")
st.title("📈 Intraday Recommender & Scanner")

# -------------------------------
# Constants & helpers
# -------------------------------
MAX_DAYS = {"1m":7, "2m":60, "5m":60, "15m":60, "30m":60, "60m":730}
FALLBACKS = {"1m":["2m","5m"], "2m":["5m","15m"], "5m":["15m","30m"], "15m":["30m","60m"], "30m":["15m","60m"], "60m":["30m","15m"]}

def cap_days(interval: str, days: int) -> int:
    return min(int(days), MAX_DAYS.get(interval, 60))

def _try(callable_fn, retries=2, delay=0.6):
    for i in range(retries+1):
        try:
            return callable_fn()
        except Exception as e:
            if i == retries: raise
            time.sleep(delay)

# -------------------------------
# Params
# -------------------------------
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
    min_confluence: int = 3

@dataclass
class StrictParams:
    rsi_long_low: float = 55.0
    rsi_long_high: float = 65.0
    rsi_short_low: float = 35.0
    rsi_short_high: float = 45.0
    vol_mult: float = 1.2
    rr_min: float = 1.8
    vwap_tolerance: float = 0.0003
    require_trend_stack: bool = True
    require_macd_zero_side: bool = True
    require_macd_rising: bool = False
    session_guard: bool = True

# -------------------------------
# Indicators
# -------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.rolling(period).mean(); rd = dn.rolling(period).mean()
    rs = ru / (rd + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series,pd.Series,pd.Series]:
    f = ema(series, fast); s = ema(series, slow)
    line = f - s; sig = ema(line, signal); hist = line - sig
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

# -------------------------------
# Cleaner
# -------------------------------
def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        wanted = {"Open","High","Low","Close","Adj Close","Volume"}
        target_level = None
        for lvl in range(df.columns.nlevels):
            if set(map(str, df.columns.get_level_values(lvl))) & wanted:
                target_level = lvl; break
        if target_level is not None:
            df.columns = df.columns.get_level_values(target_level)
        else:
            df.columns = ["_".join([str(x) for x in tup if x is not None]) for tup in df.columns]
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close":"Close"})
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if not keep: return pd.DataFrame()
    df = df[keep].copy()
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    try:
        if getattr(df.index, "tz", None) is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")
    except Exception: pass
    return df.dropna(subset=[c for c in ["Open","High","Low","Close"] if c in df.columns])

# -------------------------------
# Cache-bypass control (must influence cache keys)
# -------------------------------
with st.sidebar:
    st.header("Data")
    symbols = st.text_input("Symbols (space-separated)", "SPY TSLL AAPL")
    interval = st.selectbox("Interval", ["1m","2m","5m","15m","30m","60m"], index=2)
    days = st.number_input("Days", 1, 730, 30)
    rth_only = st.checkbox("RTH only (09:30–16:00 NY)", True)
    bypass_cache = st.checkbox("Bypass cache (force fresh data)", False)
    cache_bust = 0 if not bypass_cache else pd.Timestamp.utcnow().value

# -------------------------------
# Cached fetchers with retries
# -------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def _fetch_download_cached(ticker: str, interval: str, days: int, cache_bust=None) -> pd.DataFrame:
    days = cap_days(interval, days)
    def _call():
        return yf.download(tickers=ticker, period=f"{days}d", interval=interval,
                           auto_adjust=False, progress=False, threads=False, repair=True)
    try:
        df = _try(_call)
    except Exception:
        df = pd.DataFrame()
    return _clean_ohlcv(df)

@st.cache_data(show_spinner=False, ttl=300)
def _fetch_history_cached(ticker: str, interval: str, days: int, prepost: bool, cache_bust=None) -> pd.DataFrame:
    days = cap_days(interval, days)
    def _call():
        return yf.Ticker(ticker).history(period=f"{days}d", interval=interval,
                                         prepost=prepost, auto_adjust=False, actions=False, repair=True)
    try:
        df = _try(_call)
    except Exception:
        df = pd.DataFrame()
    return _clean_ohlcv(df)

def fetch_intraday(ticker: str, interval: str="5m", days: int=10, rth_only: bool=False, cache_bust=None) -> pd.DataFrame:
    def apply_rth(dfi: pd.DataFrame) -> pd.DataFrame:
        if dfi.empty: return dfi
        try: return dfi.between_time("09:30","16:00")
        except Exception: return dfi

    base = _fetch_download_cached(ticker, interval, days, cache_bust)
    if base.empty:
        base = _fetch_history_cached(ticker, interval, days, prepost=True, cache_bust=cache_bust)
    if base.empty:
        base = _fetch_history_cached(ticker, interval, days, prepost=False, cache_bust=cache_bust)

    dfr = apply_rth(base) if rth_only else base.copy()
    if not dfr.empty: return dfr
    if rth_only and not base.empty: return base

    for d_try in [min(cap_days(interval, days), 30), 14, 7, 5, 3]:
        tmp = _fetch_download_cached(ticker, interval, d_try, cache_bust)
        if tmp.empty: tmp = _fetch_history_cached(ticker, interval, d_try, prepost=True, cache_bust=cache_bust)
        if tmp.empty: tmp = _fetch_history_cached(ticker, interval, d_try, prepost=False, cache_bust=cache_bust)
        tmp = apply_rth(tmp) if rth_only else tmp
        if not tmp.empty: return tmp

    for iv in FALLBACKS.get(interval, []):
        tmp = _fetch_download_cached(ticker, iv, days, cache_bust)
        if tmp.empty: tmp = _fetch_history_cached(ticker, iv, days, prepost=True, cache_bust=cache_bust)
        if tmp.empty: tmp = _fetch_history_cached(ticker, iv, days, prepost=False, cache_bust=cache_bust)
        tmp = apply_rth(tmp) if rth_only else tmp
        if not tmp.empty: return tmp

    return pd.DataFrame()

# -------------------------------
# Indicators table
# -------------------------------
def compute_indicators(df: pd.DataFrame, p: SignalParams) -> pd.DataFrame:
    out = df.copy()
    out["EMA_fast"] = ema(out["Close"], p.ema_fast)
    out["EMA_slow"] = ema(out["Close"], p.ema_slow)
    out["RSI"] = rsi(out["Close"], p.rsi_period)
    m_line, m_sig, m_hist = macd(out["Close"], p.macd_fast, p.macd_slow, p.macd_signal)
    out["MACD"] = m_line; out["MACD_signal"] = m_sig; out["MACD_hist"] = m_hist
    out["ATR"] = atr(out, p.atr_period)
    out["VWAP"] = vwap(out)
    out["VolSMA20"] = out["Volume"].rolling(20).mean()
    return out

# -------------------------------
# Signal engine
# -------------------------------
def _pos_size_long(entry: float, stop: float, equity: float, risk_pct: float) -> int:
    risk_amt = equity * (risk_pct/100.0)
    per_share = max(entry - stop, 1e-9)
    return max(int(risk_amt // per_share), 0)

def _pos_size_short(entry: float, stop: float, equity: float, risk_pct: float) -> int:
    risk_amt = equity * (risk_pct/100.0)
    per_share = max(stop - entry, 1e-9)
    return max(int(risk_amt // per_share), 0)

def generate_latest_signal(df: pd.DataFrame, p: SignalParams, strict: bool=False, sp: Optional[StrictParams]=None) -> Optional[dict]:
    need = max(p.ema_slow, p.rsi_period, p.macd_slow, p.atr_period, 20) + 5
    if len(df) < need: return None
    if strict and sp is None: sp = StrictParams()

    row = df.iloc[-2]  # last completed bar
    prev = df.iloc[-3]
    close = float(row["Close"]); emaf = float(row["EMA_fast"]); emas = float(row["EMA_slow"])
    rsi_v = float(row["RSI"]); macd_h = float(row["MACD_hist"]); macd_hp = float(prev["MACD_hist"])
    vwap_v = float(row["VWAP"]); atr_v = float(row["ATR"])
    vol = float(row["Volume"]); vol_sma = float(row.get("VolSMA20", np.nan))

    if strict:
        # RTH 1st/last 30m guard
        if sp.session_guard and getattr(df.index, "tz", None) is not None:
            t = df.index[-2].tz_convert("America/New_York").time()
            if not (t >= pd.Timestamp("10:00").time() and t <= pd.Timestamp("15:30").time()):
                return None

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
        if side is None: return None

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

        if rr < sp.rr_min: return None

        return {
            "timestamp": df.index[-2], "side": side,
            "entry": round(entry, 4), "stop": round(stop, 4), "target": round(target, 4),
            "atr": round(atr_v, 4), "rsi": round(rsi_v, 2),
            "macd_hist": round(macd_h, 4), "vwap": round(vwap_v, 4),
            "rr_ratio": round(float(rr), 2), "position_size": int(size),
        }

    # non-strict confluence
    condL = int(emaf > emas) + int(50 <= rsi_v <= 70) + int((macd_h > 0) and (macd_hp <= 0)) + int(close > vwap_v)
    condS = int(emaf < emas) + int(30 <= rsi_v <= 50) + int((macd_h < 0) and (macd_hp >= 0)) + int(close < vwap_v)
    side = None
    if condL >= p.min_confluence and condL > condS: side = "LONG"
    elif condS >= p.min_confluence and condS > condL: side = "SHORT"
    if side is None: return None

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
        "timestamp": df.index[-2], "side": side,
        "entry": round(entry,4), "stop": round(stop,4), "target": round(target,4),
        "atr": round(float(atr_v),4), "rsi": round(float(rsi_v),2),
        "macd_hist": round(float(macd_h),4), "vwap": round(float(vwap_v),4),
        "rr_ratio": round(float(rr),2), "position_size": int(size),
    }

# -------------------------------
# S&P 500 (auto) + cache
# -------------------------------
@st.cache_data(show_spinner=True, ttl=3600)
def load_sp500() -> List[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df_sp = tables[0]
        syms = [s.replace(".", "-").strip().upper() for s in df_sp["Symbol"].astype(str).tolist() if s.strip()]
        # de-dup keep order
        seen = set(); out = []
        for s in syms:
            if s not in seen:
                out.append(s); seen.add(s)
        return out
    except Exception:
        fallback = "AAPL MSFT NVDA AMZN GOOGL META BRK-B JPM JNJ UNH XOM AVGO PEP LLY V MA PG COST HD MRK ABBV KO PFE BAC NFLX CRM ORCL INTC TMO MCD CSCO VZ ADP WMT"
        return [s.strip().upper() for s in fallback.split()]

# -------------------------------
# Sidebar (filters, presets, risk)
# -------------------------------
st.sidebar.divider()
st.sidebar.header("Presets")
preset = st.sidebar.selectbox("🎚️ Preset Strategy", ["Custom","Conservative","Balanced","Aggressive"])
presets = {
    "Conservative": dict(strict=True,  volume_mult=1.2,  rr_min=1.8,  vwap_tol=0.0002, rsi_long=(55,65), rsi_short=(35,45), macd_slope=True),
    "Balanced":     dict(strict=True,  volume_mult=1.1,  rr_min=1.6,  vwap_tol=0.0003, rsi_long=(53,67), rsi_short=(33,47), macd_slope=False),
    "Aggressive":   dict(strict=True,  volume_mult=1.05, rr_min=1.5,  vwap_tol=0.0005, rsi_long=(50,70), rsi_short=(30,50), macd_slope=False),
}
cfg = presets.get(preset, {})

st.sidebar.divider()
st.sidebar.header("Filters")
strict_mode = st.sidebar.checkbox("Strict mode", value=cfg.get("strict", False))
vol_mult = st.sidebar.slider("Volume × SMA20 ≥", 1.0, 2.0, cfg.get("volume_mult", 1.1), 0.05)
rr_min = st.sidebar.slider("Min RR", 1.2, 3.0, cfg.get("rr_min", 1.6), 0.1)
vwap_tol_bps = st.sidebar.slider("VWAP tolerance (bps)", 0, 20, int(cfg.get("vwap_tol", 0.0003)*10000))
vwap_tol = vwap_tol_bps / 10000.0
macd_slope = st.sidebar.checkbox("Require MACD hist rising/falling", value=cfg.get("macd_slope", False))
require_stack = st.sidebar.checkbox("Require Close>EMA20>EMA50", True)
session_guard = st.sidebar.checkbox("Skip 1st/last 30m if RTH", True)

rsi_long_min, rsi_long_max = st.sidebar.slider("RSI Long band", 40, 80, cfg.get("rsi_long", (53,67)))
rsi_short_min, rsi_short_max = st.sidebar.slider("RSI Short band", 20, 60, cfg.get("rsi_short", (33,47)))

st.sidebar.divider()
st.sidebar.header("Risk")
equity = st.sidebar.number_input("Account equity", 1000, 5_000_000, 25_000, 500)
risk_pct = st.sidebar.slider("Risk % per trade", 0.1, 5.0, 1.0, 0.1)
min_conf = st.sidebar.slider("Non-strict: min agreeing checks", 2, 5, 3)

st.sidebar.divider()
show_debug = st.sidebar.checkbox("🔧 Show raw data debug", False)
run_btn = st.sidebar.button("Run now")

# Bundle params
symbols_list = [s.strip().upper() for s in symbols.split() if s.strip()]
days_capped = cap_days(interval, days)
P = SignalParams(equity=float(equity), risk_pct=float(risk_pct), min_confluence=int(min_conf))
SP = StrictParams(
    rsi_long_low=float(rsi_long_min), rsi_long_high=float(rsi_long_max),
    rsi_short_low=float(rsi_short_min), rsi_short_high=float(rsi_short_max),
    vol_mult=float(vol_mult), rr_min=float(rr_min), vwap_tolerance=float(vwap_tol),
    require_trend_stack=bool(require_stack), require_macd_zero_side=True,
    require_macd_rising=bool(macd_slope), session_guard=bool(session_guard)
)

# -------------------------------
# Environment debug (to match cloud vs local)
# -------------------------------
with st.expander("Environment (debug)", expanded=False):
    st.write({
        "Python": tuple(__import__("sys").version.split()[0:1])[0],
        "streamlit": __import__("streamlit").__version__,
        "yfinance": __import__("yfinance").__version__,
        "pandas": __import__("pandas").__version__,
        "numpy": __import__("numpy").__version__,
        "scipy": __import__("scipy").__version__,
        "Server UTC now": pd.Timestamp.utcnow(),
        "NY time now": pd.Timestamp.utcnow().tz_localize("UTC").tz_convert("America/New_York"),
        "Interval": interval, "Days requested": int(days), "Days capped": days_capped,
        "RTH only": rth_only, "Strict": strict_mode, "Bypass cache": bypass_cache
    })

# -------------------------------
# Debug fetch preview
# -------------------------------
if show_debug and symbols_list:
    sym0 = symbols_list[0]
    st.info(f"Debug fetch for {sym0} ({interval}, {days_capped}d, RTH={'ON' if rth_only else 'OFF'})")
    dbg = fetch_intraday(sym0, interval=interval, days=days_capped, rth_only=bool(rth_only), cache_bust=cache_bust)
    st.write("Shape:", dbg.shape)
    st.write("Columns:", list(dbg.columns))
    st.dataframe(dbg.tail(12), width='stretch')

# -------------------------------
# Recommendations (current symbols)
# -------------------------------
if run_btn:
    rec_rows = []
    for sym in symbols_list:
        raw = fetch_intraday(sym, interval=interval, days=days_capped, rth_only=bool(rth_only), cache_bust=cache_bust)
        if raw.empty:
            rec_rows.append(dict(Ticker=sym, **{k:np.nan for k in ["Time(NY)","Side","Entry","Stop","Target","ATR","RSI","MACD_hist","VWAP","RR","PositionSize"]}))
            continue
        dfi = compute_indicators(raw, P).dropna().copy()
        sig = generate_latest_signal(dfi, P, strict=bool(strict_mode), sp=SP if strict_mode else None)
        if sig is None:
            rec_rows.append(dict(Ticker=sym, Time_NY=dfi.index[-2], Side="NONE",
                                 Entry=np.nan, Stop=np.nan, Target=np.nan,
                                 ATR=dfi["ATR"].iloc[-2], RSI=dfi["RSI"].iloc[-2],
                                 MACD_hist=dfi["MACD_hist"].iloc[-2], VWAP=dfi["VWAP"].iloc[-2],
                                 RR=np.nan, PositionSize=0))
        else:
            ts = sig["timestamp"]
            try: ts = ts.tz_convert("America/New_York")
            except Exception: pass
            rec_rows.append({
                "Ticker": sym, "Time(NY)": ts, "Side": sig["side"],
                "Entry": sig["entry"], "Stop": sig["stop"], "Target": sig["target"],
                "ATR": sig["atr"], "RSI": sig["rsi"], "MACD_hist": sig["macd_hist"], "VWAP": sig["vwap"],
                "RR": sig["rr_ratio"], "PositionSize": sig["position_size"]
            })

    st.subheader("Recommendations (latest completed bar)")
    rec_df = pd.DataFrame(rec_rows)
    if not rec_df.empty:
        cols = ["Ticker","Time(NY)","Side","Entry","Stop","Target","RR","PositionSize","ATR","RSI","MACD_hist","VWAP"]
        rec_df = rec_df[[c for c in cols if c in rec_df.columns]]
        st.dataframe(rec_df, width='stretch')
    else:
        st.info("No rows. Try RTH OFF first or relax strict filters / lower agreeing checks.")

    st.divider()
    st.subheader("Details per symbol")
    for sym in symbols_list:
        st.markdown(f"**{sym}**")
        raw = fetch_intraday(sym, interval=interval, days=days_capped, rth_only=bool(rth_only), cache_bust=cache_bust)
        if raw.empty:
            st.warning(f"No data for {sym}")
            continue
        dfi = compute_indicators(raw, P).dropna().copy()
        st.dataframe(dfi.tail(12), width='stretch')
        st.line_chart(dfi["Close"])

# -------------------------------
# Scanner (S&P500 auto + custom)
# -------------------------------
st.divider()
st.header("🔎 Scan for Opportunities")

scan_col1, scan_col2 = st.columns([2,1])
with scan_col1:
    universe_choice = st.selectbox(
        "Universe",
        [
            "S&P 500 (auto)",
            "Top Tech (AAPL MSFT NVDA AMZN GOOGL META AMD TSLA)",
            "Index ETFs (SPY QQQ IWM DIA TLT HYG XLF XLK XLE XLY)",
            "Mega Liquids (SPY AAPL MSFT NVDA TSLA AMD AMZN META GOOGL QQQ)",
            "Levered ETFs (TSLL TQQQ SOXL LABU SQQQ SOXS UVXY SVXY)",
            "Custom (type below)"
        ],
        index=0
    )
with scan_col2:
    show_only_actionable = st.checkbox("Only actionable (Side ≠ NONE)", True)

custom_universe = st.text_input("Custom symbols (space-separated)", "SPY TSLL AAPL")

def parse_universe(choice: str, custom: str) -> List[str]:
    if choice == "S&P 500 (auto)":
        syms = load_sp500()
    else:
        mapping = {
            "Top Tech (AAPL MSFT NVDA AMZN GOOGL META AMD TSLA)":
                "AAPL MSFT NVDA AMZN GOOGL META AMD TSLA",
            "Index ETFs (SPY QQQ IWM DIA TLT HYG XLF XLK XLE XLY)":
                "SPY QQQ IWM DIA TLT HYG XLF XLK XLE XLY",
            "Mega Liquids (SPY AAPL MSFT NVDA TSLA AMD AMZN META GOOGL QQQ)":
                "SPY AAPL MSFT NVDA TSLA AMD AMZN META GOOGL QQQ",
            "Levered ETFs (TSLL TQQQ SOXL LABU SQQQ SOXS UVXY SVXY)":
                "TSLL TQQQ SOXL LABU SQQQ SOXS UVXY SVXY",
            "Custom (type below)":
                custom
        }
        syms = [s.strip().upper() for s in mapping.get(choice, custom).split() if s.strip()]
    # de-dup keep order
    seen = set(); out = []
    for s in syms:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

scan_syms = parse_universe(universe_choice, custom_universe)
max_scan = st.slider("Max tickers to scan", 50, 500, 150, 25)
scan_syms = scan_syms[:max_scan]
st.caption(f"Scanning {len(scan_syms)} tickers… Adjust with the slider above.")

scan_save_csv = st.text_input("Save scan results CSV (optional path)", "")
scan_btn = st.button("Run Scan")

def scan_universe(symbols: List[str], interval: str, days: int, rth_only: bool,
                  P: SignalParams, strict_mode: bool, SP: StrictParams, cache_bust=None) -> pd.DataFrame:
    rows = []
    n = len(symbols)
    prog = st.progress(0.0, text=f"Scanning {n} symbols…")
    for i, sym in enumerate(symbols, 1):
        try:
            raw = fetch_intraday(sym, interval=interval, days=days, rth_only=rth_only, cache_bust=cache_bust)
            if raw.empty:
                rows.append({"Ticker": sym, "Side": "NONE", "Reason": "No data"})
            else:
                dfi = compute_indicators(raw, P).dropna().copy()
                sig = generate_latest_signal(dfi, P, strict=strict_mode, sp=SP if strict_mode else None)
                if sig is None:
                    rows.append({
                        "Ticker": sym, "Side": "NONE",
                        "RSI": round(float(dfi["RSI"].iloc[-2]),2),
                        "MACD_hist": round(float(dfi["MACD_hist"].iloc[-2]),4),
                        "VWAP": round(float(dfi["VWAP"].iloc[-2]),4),
                        "ATR": round(float(dfi["ATR"].iloc[-2]),4),
                    })
                else:
                    ts = sig["timestamp"]
                    try: ts = ts.tz_convert("America/New_York")
                    except Exception: pass
                    rows.append({
                        "Ticker": sym, "Time(NY)": ts, "Side": sig["side"],
                        "Entry": sig["entry"], "Stop": sig["stop"], "Target": sig["target"],
                        "RR": sig["rr_ratio"], "PositionSize": sig["position_size"],
                        "ATR": sig["atr"], "RSI": sig["rsi"], "MACD_hist": sig["macd_hist"], "VWAP": sig["vwap"]
                    })
        except Exception as e:
            rows.append({"Ticker": sym, "Side": "ERROR", "Reason": str(e)})
        prog.progress(i / max(n,1), text=f"Scanning {n} symbols… ({i}/{n})")
    return pd.DataFrame(rows)

if scan_btn:
    with st.spinner("Sweeping universe…"):
        scan_df = scan_universe(scan_syms, interval=interval, days=days_capped, rth_only=bool(rth_only),
                                P=P, strict_mode=bool(strict_mode), SP=SP, cache_bust=cache_bust)

    st.subheader("Scan Results")
    if show_only_actionable and not scan_df.empty:
        actionable = scan_df[scan_df["Side"].isin(["LONG","SHORT"])].copy()
        if actionable.empty:
            st.info("No actionable signals found with current filters.")
        else:
            if "RR" in actionable.columns:
                actionable = actionable.sort_values(by=["RR","PositionSize"], ascending=[False, False])
            st.dataframe(actionable, width='stretch')
    else:
        st.dataframe(scan_df, width='stretch')

    if scan_save_csv.strip():
        try:
            scan_df.to_csv(scan_save_csv, index=False)
            st.success(f"Saved scan → {scan_save_csv}")
        except Exception as e:
            st.warning(f"Could not save CSV: {e}")
