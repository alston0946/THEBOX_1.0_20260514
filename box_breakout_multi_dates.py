# -*- coding: utf-8 -*-
"""
箱体突破扫描 V3 - Tushare 快速版 / 非淘汰式评分版

核心变化：
1. 保留 Tushare 快速取数结构：每只股票只取一次数据，再按多个 TARGET_DATES 本地切片。
2. BOX_WINDOWS = 40~120，步长 1。
3. 不再用箱体宽度做硬淘汰；大箱体、小箱体都保留，只做分类、评分和备注。
4. 删除箱体重心斜率过滤与评分。
5. valid_windows 不再作为硬淘汰条件；只要有一个窗口通过核心突破条件即可输出，窗口数量只做加分。
6. 箱顶检测改为 high + close 双重等高压力测试：
   - 用 high 判断是否触碰箱顶；
   - 同时统计箱顶触碰段的 high 等高程度和 close 等高程度；
   - high 与 close 都等高时给更高评分；
   - high 等高但 close 分歧时保留候选，但在备注中提示。
7. 箱底检测不再硬淘汰，只输出触碰日期、代表低点价格，并参与支撑确认评分。
8. 最终输出 S/A/B/C/D 等级，适合先保留候选，再人工复盘。
9. 严格要求 signal_date 与 target_date 是同一天；若目标日不是交易日或个股当日无数据，则不输出信号。
"""

import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tushare as ts


# =========================
# 清理代理
# =========================
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)


# =========================
# 仓库路径
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CODE_FILE = os.path.join(DATA_DIR, "a_share_codes_for_akshare.csv")
BELOW_8B_FILE = os.path.join(DATA_DIR, "a_share_below_8b.csv")
ST_FILE = os.path.join(DATA_DIR, "st_stocks.csv")


# =========================
# 环境变量 / 运行参数
# =========================
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()

# GitHub Actions 可通过环境变量传入：
# TARGET_DATES=20260513 或 TARGET_DATES=20260512,20260513
# 未传入 TARGET_DATES 时，默认使用北京时间今天。
START_DATE = os.getenv("START_DATE", "20240301").strip()
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
TEST_LIMIT = None if not os.getenv("TEST_LIMIT") else int(os.getenv("TEST_LIMIT"))
BATCH_START = int(os.getenv("BATCH_START", "0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10000"))
SLEEP_SEC = float(os.getenv("SLEEP_SEC", "0.03"))


# =========================
# 日期处理
# =========================
def get_today_cn_str() -> str:
    return (datetime.utcnow() + timedelta(hours=8)).strftime("%Y%m%d")


def get_target_dates():
    """
    优先读取环境变量 TARGET_DATES，例如：
    TARGET_DATES=20260513
    TARGET_DATES=20260512,20260513
    否则默认使用北京时间今天。
    """
    raw = os.getenv("TARGET_DATES", "").strip()
    if raw:
        items = []
        for part in raw.replace(";", ",").replace("\n", ",").split(","):
            s = part.strip()
            if s:
                items.append(s)
    else:
        items = [get_today_cn_str()]

    target_dates = sorted({x for x in items if x.isdigit() and len(x) == 8})
    if not target_dates:
        raise ValueError("TARGET_DATES 为空，且未能生成默认日期。")
    return target_dates


# =========================
# 箱体策略参数 V3
# =========================
MIN_BOX_DAYS = 40
MAX_BOX_DAYS = 120
BOX_STEP = 1
BOX_WINDOWS = list(range(MIN_BOX_DAYS, MAX_BOX_DAYS + 1, BOX_STEP))

# 箱体宽度不作为硬条件，只用于分类和轻微评分
NARROW_BOX_WIDTH_PCT = 0.08
STANDARD_BOX_WIDTH_PCT = 0.25
WIDE_BOX_WIDTH_PCT = 0.45

# 箱体内收盘占比：宽松硬条件，主要防止完全乱波动
MIN_INBOX_RATIO_HARD = 0.55
LOWER_BUFFER = 0.015
UPPER_BUFFER = 0.015

# 明显跌破下沿次数：不做硬淘汰，评分扣分用
LOWER_BREAK_BUFFER = 0.015

# 突破条件：核心硬条件
BREAKOUT_PCT = 0.003       # 目标日收盘突破箱顶 0.3%
YDAY_ALLOW_PCT = 0.010     # 昨日允许轻微突破，但不能已经明显突破
NEW_HIGH_BUFFER = 0.001    # 今日收盘要略高于箱体期内最高收盘

# 箱顶/箱底触碰设置
MIN_TOP_TOUCH_SEGMENTS_HARD = 1  # 至少要有 1 段箱顶测试，否则没有可验证箱顶
UPPER_TOUCH_BAND = 0.030         # high 接近箱顶的下容忍
UPPER_CLOSE_ALLOW_PCT = 0.030    # 箱体期内箱顶测试日，close 不应过度站上箱顶
LOWER_TOUCH_BAND = 0.030         # low 接近箱底的上容忍
LOWER_CLOSE_ALLOW_PCT = 0.030    # 箱底测试日，close 允许略低于箱底
MIN_TOUCH_GAP = 5                # 两个触碰段之间至少间隔多少个非触碰交易日

# 箱顶等高偏差评分阈值：不再硬淘汰
# high 偏差用于判断盘中多次冲击箱顶是否等高。
TOP_SPREAD_STRONG_PCT = 0.020
TOP_SPREAD_GOOD_PCT = 0.040
TOP_SPREAD_OK_PCT = 0.070
TOP_SPREAD_WEAK_PCT = 0.100

# close 偏差用于判断多次箱顶测试时，收盘价是否也处在相近位置。
# high 与 close 都等高，说明箱顶压力区更可信，评分更高。
TOP_CLOSE_SPREAD_STRONG_PCT = 0.020
TOP_CLOSE_SPREAD_GOOD_PCT = 0.040
TOP_CLOSE_SPREAD_OK_PCT = 0.070
TOP_CLOSE_SPREAD_WEAK_PCT = 0.100

# 突破前位置评分用，不做硬淘汰
RECENT_WEAK_DAYS = 7

# 趋势评分均线
MA_SHORT = 20
MA_LONG = 60

# valid_windows 只做评分，不做硬淘汰
WINDOW_SUPPORT_FULL_COUNT = 5


# =========================
# CSV / 股票池工具函数
# =========================
def read_csv_safely(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, dtype=str)
        except Exception as e:
            last_err = e
    raise last_err


def normalize_to_6digits(x) -> str:
    s = str(x).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 6:
        return digits
    if len(digits) > 6:
        return digits[-6:]
    if len(digits) > 0:
        return digits.zfill(6)
    return ""


def code6_to_ts_code(code6: str) -> str:
    if code6.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"{code6}.SH"
    return f"{code6}.SZ"


def load_code_set_from_csv(path: str) -> set:
    df = read_csv_safely(path)
    preferred_cols = ["ticker", "symbol", "code", "code6", "股票代码", "证券代码", "ts_code"]
    code_col = None
    for c in preferred_cols:
        if c in df.columns:
            code_col = c
            break
    if code_col is None:
        code_col = df.columns[0]
    return set(df[code_col].map(normalize_to_6digits).dropna().tolist())


def load_universe_from_csv():
    df = read_csv_safely(CODE_FILE)

    if "ticker" in df.columns:
        ticker_col = "ticker"
    elif "symbol" in df.columns:
        ticker_col = "symbol"
    else:
        raise ValueError(f"代码文件缺少 ticker/symbol 列，实际列名: {list(df.columns)}")

    name_col = "secShortName" if "secShortName" in df.columns else None

    out = df.copy()
    out["ticker"] = out[ticker_col].map(normalize_to_6digits)
    out = out[out["ticker"].str.len() == 6].copy()

    if name_col:
        out["name"] = out[name_col].astype(str).str.strip()
    else:
        out["name"] = out["ticker"]

    before_filter = len(out)

    st_codes = load_code_set_from_csv(ST_FILE)
    below_8b_codes = load_code_set_from_csv(BELOW_8B_FILE)

    out["is_st_file"] = out["ticker"].isin(st_codes)
    out["is_st_name"] = out["name"].astype(str).str.contains(r"ST|退", case=False, regex=True, na=False)
    out["is_below_8b"] = out["ticker"].isin(below_8b_codes)

    filtered_out = out[(out["is_st_file"]) | (out["is_st_name"]) | (out["is_below_8b"])].copy()
    universe = out[(~out["is_st_file"]) & (~out["is_st_name"]) & (~out["is_below_8b"])].copy()

    universe["ts_code"] = universe["ticker"].map(code6_to_ts_code)
    filtered_out["ts_code"] = filtered_out["ticker"].map(code6_to_ts_code)

    universe = universe.drop_duplicates("ticker").reset_index(drop=True)
    filtered_out = filtered_out.drop_duplicates("ticker").reset_index(drop=True)

    print(
        f"股票池过滤统计：原始 {before_filter} 只 | "
        f"剔除 ST代码名单 {int(out['is_st_file'].sum())} 只 | "
        f"剔除 ST名称兜底 {int((~out['is_st_file'] & out['is_st_name']).sum())} 只 | "
        f"剔除 80亿以下 {int((~out['is_st_file'] & ~out['is_st_name'] & out['is_below_8b']).sum())} 只 | "
        f"剩余 {len(universe)} 只"
    )

    if TEST_LIMIT is not None:
        universe = universe.head(TEST_LIMIT).copy()
    else:
        universe = universe.iloc[BATCH_START:BATCH_START + BATCH_SIZE].copy()

    return universe[["ticker", "ts_code", "name"]].reset_index(drop=True), filtered_out


# =========================
# Tushare 取数
# =========================
def fetch_tushare_daily_with_retry(pro, ts_code: str, start_date: str, end_date: str, max_retry: int = 3):
    last_err = None
    for attempt in range(max_retry):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                return df
            last_err = "empty dataframe"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        time.sleep(0.6 + attempt * 0.6)
    raise RuntimeError(str(last_err))


def standardize_tushare_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    required_cols = ["trade_date", "open", "high", "low", "close", "vol"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Tushare daily 缺少字段: {missing}")

    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["vol"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce") if "amount" in out.columns else np.nan

    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"]).copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out[["date", "open", "high", "low", "close", "volume", "amount"]]


# =========================
# 技术工具函数
# =========================
def calc_ma(df, windows=(20, 60)):
    out = df.copy()
    for w in windows:
        out[f"ma{w}"] = out["close"].rolling(w).mean()
    return out


def calc_box_features(close_arr: np.ndarray):
    """用收盘价 10% / 90% 分位数估计候选箱体上下沿。"""
    lower = float(np.quantile(close_arr, 0.10))
    upper = float(np.quantile(close_arr, 0.90))
    return lower, upper


def count_touch_segments(mask: np.ndarray, min_gap: int = 5):
    """连续多天触碰只算一段；两个触碰段之间间隔过近则合并。"""
    mask = np.asarray(mask, dtype=bool)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return 0, []

    runs = []
    start = int(idxs[0])
    prev = int(idxs[0])

    for idx in idxs[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = idx
            prev = idx
    runs.append((start, prev))

    merged = []
    for run_start, run_end in runs:
        if not merged:
            merged.append((run_start, run_end))
            continue
        last_start, last_end = merged[-1]
        gap = run_start - last_end - 1
        if gap < min_gap:
            merged[-1] = (last_start, run_end)
        else:
            merged.append((run_start, run_end))

    return len(merged), merged


def summarize_touch_segments(box_dates, price_arr, segments, mode="upper"):
    """
    汇总触碰段代表日期和代表价格。
    mode="upper"：每段取最高价作为箱顶测试价。
    mode="lower"：每段取最低价作为箱底测试价。
    """
    dates = []
    prices = []
    ranges = []

    if not segments:
        return {"dates_str": "", "prices_str": "", "ranges_str": "", "prices": []}

    dates_arr = pd.to_datetime(pd.Series(box_dates)).reset_index(drop=True)
    price_arr = np.asarray(price_arr, dtype=float)

    for start, end in segments:
        seg_prices = price_arr[start:end + 1]
        if len(seg_prices) == 0:
            continue

        offset = int(np.argmin(seg_prices)) if mode == "lower" else int(np.argmax(seg_prices))
        ref_idx = start + offset
        ref_date = dates_arr.iloc[ref_idx].strftime("%Y-%m-%d")
        ref_price = float(price_arr[ref_idx])

        start_date = dates_arr.iloc[start].strftime("%Y-%m-%d")
        end_date = dates_arr.iloc[end].strftime("%Y-%m-%d")
        range_str = start_date if start_date == end_date else f"{start_date}~{end_date}"

        dates.append(f"{ref_date}@{ref_price:.2f}")
        prices.append(ref_price)
        ranges.append(range_str)

    return {
        "dates_str": "|".join(dates),
        "prices_str": "|".join(f"{x:.4f}" for x in prices),
        "ranges_str": "|".join(ranges),
        "prices": prices,
    }


def calc_spread_pct(prices, reference_price):
    if prices is None or len(prices) <= 1:
        return 0.0
    if reference_price is None or reference_price <= 0:
        return np.nan
    return float((max(prices) - min(prices)) / reference_price)


def classify_box_width(width_pct):
    if width_pct < NARROW_BOX_WIDTH_PCT:
        return "窄箱体"
    if width_pct < STANDARD_BOX_WIDTH_PCT:
        return "标准箱体"
    if width_pct < WIDE_BOX_WIDTH_PCT:
        return "宽箱体"
    return "超宽箱体"


def classify_spread_level(spread_pct, strong, good, ok, weak):
    if pd.isna(spread_pct):
        return "未知"
    if spread_pct <= strong:
        return "强等高"
    if spread_pct <= good:
        return "较等高"
    if spread_pct <= ok:
        return "一般等高"
    if spread_pct <= weak:
        return "弱等高"
    return "不等高"


def classify_top_quality(high_spread_pct, close_spread_pct, touch_segments):
    """
    箱顶质量备注：同时考虑 high 是否等高、close 是否等高。
    high 等高说明多次冲击同一压力区；close 也等高说明收盘承压/蓄势位置也一致。
    """
    if touch_segments < 2:
        return "箱顶测试不足"

    high_level = classify_spread_level(
        high_spread_pct,
        TOP_SPREAD_STRONG_PCT,
        TOP_SPREAD_GOOD_PCT,
        TOP_SPREAD_OK_PCT,
        TOP_SPREAD_WEAK_PCT,
    )
    close_level = classify_spread_level(
        close_spread_pct,
        TOP_CLOSE_SPREAD_STRONG_PCT,
        TOP_CLOSE_SPREAD_GOOD_PCT,
        TOP_CLOSE_SPREAD_OK_PCT,
        TOP_CLOSE_SPREAD_WEAK_PCT,
    )

    if high_level == "强等高" and close_level == "强等高":
        return "高点+收盘双强等高箱顶"
    if high_level in ["强等高", "较等高"] and close_level in ["强等高", "较等高"]:
        return "高点+收盘双等高箱顶"
    if high_level in ["强等高", "较等高"] and close_level in ["一般等高", "弱等高"]:
        return "高点等高但收盘有分歧"
    if high_level in ["一般等高", "弱等高"] and close_level in ["强等高", "较等高", "一般等高"]:
        return "箱顶一般等高，收盘较一致"
    if high_level in ["强等高", "较等高", "一般等高", "弱等高"]:
        return "高点箱顶尚可，收盘分歧较大"
    return "箱顶不够清晰"


def calc_trend_score(df):
    """
    趋势评分，总分 10。
    只评分，不做硬过滤。
    """
    if len(df) < MA_LONG + 10:
        return 0.0, "bars_not_enough_for_trend_score"

    tmp = calc_ma(df, windows=(MA_SHORT, MA_LONG))
    row = tmp.iloc[-1]
    ma20_col = f"ma{MA_SHORT}"
    ma60_col = f"ma{MA_LONG}"

    if pd.isna(row[ma20_col]) or pd.isna(row[ma60_col]):
        return 0.0, "ma_nan"

    score = 0.0
    notes = []

    if row["close"] > row[ma20_col]:
        score += 4
        notes.append("close_gt_ma20")
    else:
        notes.append("close_le_ma20")

    if row[ma20_col] > row[ma60_col]:
        score += 3
        notes.append("ma20_gt_ma60")
    else:
        notes.append("ma20_le_ma60")

    ma20_now = tmp[ma20_col].iloc[-1]
    ma20_5ago = tmp[ma20_col].iloc[-6] if len(tmp) >= 6 else np.nan
    if pd.notna(ma20_now) and pd.notna(ma20_5ago) and ma20_now >= ma20_5ago:
        score += 2
        notes.append("ma20_5d_rising")
    else:
        notes.append("ma20_5d_not_rising")

    ma60_now = tmp[ma60_col].iloc[-1]
    ma60_10ago = tmp[ma60_col].iloc[-11] if len(tmp) >= 11 else np.nan
    if pd.notna(ma60_now) and pd.notna(ma60_10ago):
        ma60_change = ma60_now / ma60_10ago - 1 if ma60_10ago > 0 else np.nan
        if pd.notna(ma60_change) and ma60_change >= -0.01:
            score += 1
            notes.append("ma60_not_obviously_down")
        else:
            notes.append("ma60_down")
    else:
        notes.append("ma60_change_nan")

    return float(score), "|".join(notes)


# =========================
# 评分函数
# =========================
def _spread_score(spread_pct, strong, good, ok, weak, scores):
    if pd.isna(spread_pct):
        return 0.0
    if spread_pct <= strong:
        return float(scores[0])
    if spread_pct <= good:
        return float(scores[1])
    if spread_pct <= ok:
        return float(scores[2])
    if spread_pct <= weak:
        return float(scores[3])
    return float(scores[4])


def score_top_quality(touch_segments, high_spread_pct, close_spread_pct):
    """
    箱顶质量 30 分。
    - 触碰段数量：最多 10 分；
    - high 等高程度：最多 12 分；
    - close 等高程度：最多 8 分。

    这样 high 和 close 同时等高会自然获得更高分；
    如果只是 high 等高但 close 分歧，仍保留候选，但分数会低一些。
    """
    if touch_segments <= 0:
        segment_score = 0
    elif touch_segments == 1:
        segment_score = 4
    elif touch_segments == 2:
        segment_score = 8
    else:
        segment_score = 10

    high_score = _spread_score(
        high_spread_pct,
        TOP_SPREAD_STRONG_PCT,
        TOP_SPREAD_GOOD_PCT,
        TOP_SPREAD_OK_PCT,
        TOP_SPREAD_WEAK_PCT,
        scores=[12, 10, 7, 4, 0],
    )
    close_score = _spread_score(
        close_spread_pct,
        TOP_CLOSE_SPREAD_STRONG_PCT,
        TOP_CLOSE_SPREAD_GOOD_PCT,
        TOP_CLOSE_SPREAD_OK_PCT,
        TOP_CLOSE_SPREAD_WEAK_PCT,
        scores=[8, 6, 4, 2, 0],
    )

    return round(min(30.0, segment_score + high_score + close_score), 4)


def score_breakout(breakout_pct_vs_top, prev_close, top_price, last_close, box_max_close, recent_gap_pct):
    """突破质量 20 分。"""
    score = 0.0
    if breakout_pct_vs_top >= BREAKOUT_PCT:
        score += 10
    score += min(4.0, max(0.0, (breakout_pct_vs_top - BREAKOUT_PCT) * 100))
    if last_close > box_max_close * (1 + NEW_HIGH_BUFFER):
        score += 3
    if prev_close <= top_price * (1 + YDAY_ALLOW_PCT):
        score += 2
    if recent_gap_pct <= 0.07:
        score += 1
    return round(min(20.0, score), 4)


def score_stability(inbox_ratio, below_lower_count, w):
    """箱体稳定性 15 分。"""
    inbox_score = min(10.0, max(0.0, (inbox_ratio - 0.55) / (0.90 - 0.55) * 10))
    below_ratio = below_lower_count / max(w, 1)
    below_score = max(0.0, 5.0 - below_ratio * 50)
    return round(inbox_score + below_score, 4)


def score_support(lower_segments, bottom_spread_pct):
    """支撑确认 10 分，不硬淘汰。"""
    if lower_segments <= 0:
        seg_score = 0
    elif lower_segments == 1:
        seg_score = 3
    elif lower_segments == 2:
        seg_score = 5
    else:
        seg_score = 6

    if lower_segments <= 1:
        spread_score = 1
    elif bottom_spread_pct <= 0.03:
        spread_score = 4
    elif bottom_spread_pct <= 0.06:
        spread_score = 3
    elif bottom_spread_pct <= 0.10:
        spread_score = 2
    else:
        spread_score = 0
    return round(min(10.0, seg_score + spread_score), 4)


def score_width(width_pct):
    """箱体宽度 5 分，不硬淘汰。"""
    # 标准箱体最高，窄/宽也保留，超宽略低。
    if width_pct < NARROW_BOX_WIDTH_PCT:
        return 3.5
    if width_pct < STANDARD_BOX_WIDTH_PCT:
        return 5.0
    if width_pct < WIDE_BOX_WIDTH_PCT:
        return 4.0
    return 2.0


def score_window_support(valid_window_count):
    """窗口支持度 10 分，不硬淘汰。"""
    return round(min(10.0, valid_window_count / WINDOW_SUPPORT_FULL_COUNT * 10), 4)


def classify_signal(final_score):
    if final_score >= 85:
        return "S级-高质量箱体突破"
    if final_score >= 75:
        return "A级-有效箱体突破"
    if final_score >= 65:
        return "B级-箱体突破候选"
    if final_score >= 55:
        return "C级-疑似突破观察"
    return "D级-低质量观察"


def build_pattern_note(row):
    notes = []
    notes.append(row.get("box_top_quality", ""))

    close_spread = row.get("box_top_touch_close_spread_pct", np.nan)
    high_spread = row.get("box_top_touch_height_spread_pct", np.nan)
    if pd.notna(close_spread) and pd.notna(high_spread):
        if high_spread <= TOP_SPREAD_GOOD_PCT * 100 and close_spread <= TOP_CLOSE_SPREAD_GOOD_PCT * 100:
            notes.append("箱顶高点与收盘价均较等高")
        elif high_spread <= TOP_SPREAD_GOOD_PCT * 100 and close_spread > TOP_CLOSE_SPREAD_OK_PCT * 100:
            notes.append("高点等高但收盘分歧偏大")

    notes.append(row.get("box_width_type", ""))

    if row.get("touch_lower_segments", 0) == 0:
        notes.append("箱底未充分确认")
    elif row.get("touch_lower_segments", 0) == 1:
        notes.append("箱底有一次支撑测试")
    elif row.get("touch_lower_segments", 0) >= 2:
        notes.append("箱底支撑测试较清晰")

    if row.get("valid_window_count", 0) == 1:
        notes.append("单窗口命中")
    elif row.get("valid_window_count", 0) >= 4:
        notes.append("多窗口支持较强")
    else:
        notes.append("多窗口支持一般")

    return " | ".join([x for x in notes if x])


# =========================
# 箱体突破核心识别
# =========================
def check_one_window(df, w, trend_score, trend_note):
    if len(df) < w + 2:
        return None

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    last_close = float(last_row["close"])
    prev_close = float(prev_row["close"])
    last_date = last_row["date"]

    # 用目标日前一天往前 w 天识别箱体，目标日只做突破判断。
    box_df = df.iloc[-(w + 1):-1].copy().reset_index(drop=True)
    if len(box_df) != w:
        return None

    box_dates = box_df["date"].values
    box_closes = box_df["close"].values.astype(float)
    box_highs = box_df["high"].values.astype(float)
    box_lows = box_df["low"].values.astype(float)

    if np.any(~np.isfinite(box_closes)):
        return None

    lower, upper = calc_box_features(box_closes)
    if lower <= 0 or upper <= lower:
        return None

    width_pct = (upper - lower) / lower
    box_width_type = classify_box_width(width_pct)

    # 宽松箱体内占比，只防止完全无箱体结构。
    inbox_low = lower * (1 - LOWER_BUFFER)
    inbox_high = upper * (1 + UPPER_BUFFER)
    inbox_mask = (box_closes >= inbox_low) & (box_closes <= inbox_high)
    inbox_ratio = float(inbox_mask.mean())
    if inbox_ratio < MIN_INBOX_RATIO_HARD:
        return None

    below_lower_mask = box_closes < lower * (1 - LOWER_BREAK_BUFFER)
    below_lower_count = int(below_lower_mask.sum())

    # 箱顶触碰：用 high 检测是否冲击箱顶，close 不应过度站上箱顶。
    upper_touch_mask = (
        (box_highs >= upper * (1 - UPPER_TOUCH_BAND))
        & (box_closes <= upper * (1 + UPPER_CLOSE_ALLOW_PCT))
    )
    touch_upper_segments, upper_segments = count_touch_segments(upper_touch_mask, min_gap=MIN_TOUCH_GAP)
    if touch_upper_segments < MIN_TOP_TOUCH_SEGMENTS_HARD:
        return None

    # 箱顶 high 等高：每段取最高价作为“冲击箱顶”的代表价格。
    upper_touch_info = summarize_touch_segments(
        box_dates=box_dates,
        price_arr=box_highs,
        segments=upper_segments,
        mode="upper",
    )
    upper_prices = upper_touch_info["prices"]
    top_price = float(np.median(upper_prices)) if upper_prices else upper
    top_spread_pct = calc_spread_pct(upper_prices, top_price)
    if pd.isna(top_spread_pct):
        top_spread_pct = 0.0

    # 箱顶 close 等高：沿用同一批触碰段，每段取最高收盘价作为代表。
    upper_close_touch_info = summarize_touch_segments(
        box_dates=box_dates,
        price_arr=box_closes,
        segments=upper_segments,
        mode="upper",
    )
    upper_close_prices = upper_close_touch_info["prices"]
    top_close_ref = float(np.median(upper_close_prices)) if upper_close_prices else upper
    top_close_spread_pct = calc_spread_pct(upper_close_prices, top_close_ref)
    if pd.isna(top_close_spread_pct):
        top_close_spread_pct = 0.0

    box_top_quality = classify_top_quality(
        high_spread_pct=top_spread_pct,
        close_spread_pct=top_close_spread_pct,
        touch_segments=touch_upper_segments,
    )

    # 箱底触碰：只做评分和输出，不硬淘汰。
    lower_touch_mask = (
        (box_lows <= lower * (1 + LOWER_TOUCH_BAND))
        & (box_closes >= lower * (1 - LOWER_CLOSE_ALLOW_PCT))
    )
    touch_lower_segments, lower_segments = count_touch_segments(lower_touch_mask, min_gap=MIN_TOUCH_GAP)
    lower_touch_info = summarize_touch_segments(
        box_dates=box_dates,
        price_arr=box_lows,
        segments=lower_segments,
        mode="lower",
    )
    lower_prices = lower_touch_info["prices"]
    bottom_price = float(np.median(lower_prices)) if lower_prices else lower
    bottom_spread_pct = calc_spread_pct(lower_prices, bottom_price)
    if pd.isna(bottom_spread_pct):
        bottom_spread_pct = 0.0

    recent_len = min(RECENT_WEAK_DAYS, len(box_closes))
    recent_max_close = float(np.max(box_closes[-recent_len:]))
    recent_gap_pct = max(0.0, (top_price - recent_max_close) / top_price) if top_price > 0 else np.nan

    # 核心突破硬条件：目标日收盘突破“箱顶检测价”。
    if not (last_close > top_price * (1 + BREAKOUT_PCT)):
        return None

    # 昨日不能已经明显突破，否则不是当天突破。
    if prev_close > top_price * (1 + YDAY_ALLOW_PCT):
        return None

    # 今日收盘要创箱体期内收盘新高。
    box_max_close = float(np.max(box_closes))
    if last_close <= box_max_close * (1 + NEW_HIGH_BUFFER):
        return None

    breakout_pct_vs_top = last_close / top_price - 1

    top_score = score_top_quality(touch_upper_segments, top_spread_pct, top_close_spread_pct)
    breakout_score = score_breakout(
        breakout_pct_vs_top=breakout_pct_vs_top,
        prev_close=prev_close,
        top_price=top_price,
        last_close=last_close,
        box_max_close=box_max_close,
        recent_gap_pct=recent_gap_pct,
    )
    stability_score = score_stability(inbox_ratio, below_lower_count, w)
    support_score = score_support(touch_lower_segments, bottom_spread_pct)
    width_score = score_width(width_pct)

    # 不含窗口支持分，窗口支持分在所有窗口扫描结束后统一计算。
    base_score = (
        top_score
        + breakout_score
        + stability_score
        + support_score
        + trend_score
        + width_score
    )

    return {
        "matched": True,
        "date": pd.to_datetime(last_date).strftime("%Y-%m-%d"),
        "window": int(w),

        "box_lower": round(lower, 4),
        "box_upper_q90": round(upper, 4),
        "box_top_price": round(top_price, 4),
        "box_bottom_price": round(bottom_price, 4),
        "box_width_pct": round(width_pct * 100, 2),
        "box_width_type": box_width_type,

        "inbox_ratio_pct": round(inbox_ratio * 100, 2),
        "below_lower_count": below_lower_count,
        "below_lower_ratio_pct": round(below_lower_count / max(w, 1) * 100, 2),

        "touch_upper_segments": int(touch_upper_segments),
        "touch_lower_segments": int(touch_lower_segments),
        "box_top_quality": box_top_quality,
        "box_top_touch_dates": upper_touch_info["dates_str"],
        "box_top_touch_ranges": upper_touch_info["ranges_str"],
        "box_top_touch_prices": upper_touch_info["prices_str"],
        "box_top_touch_height_spread_pct": round(top_spread_pct * 100, 2),
        "box_top_touch_close_dates": upper_close_touch_info["dates_str"],
        "box_top_touch_close_prices": upper_close_touch_info["prices_str"],
        "box_top_touch_close_spread_pct": round(top_close_spread_pct * 100, 2),
        "box_bottom_touch_dates": lower_touch_info["dates_str"],
        "box_bottom_touch_ranges": lower_touch_info["ranges_str"],
        "box_bottom_touch_prices": lower_touch_info["prices_str"],
        "box_bottom_touch_height_spread_pct": round(bottom_spread_pct * 100, 2),

        "recent_gap_pct": round(recent_gap_pct * 100, 2) if pd.notna(recent_gap_pct) else np.nan,
        "prev_close": round(prev_close, 4),
        "last_close": round(last_close, 4),
        "breakout_pct_vs_top": round(breakout_pct_vs_top * 100, 2),
        "box_max_close": round(box_max_close, 4),

        "top_score": round(top_score, 4),
        "breakout_score": round(breakout_score, 4),
        "stability_score": round(stability_score, 4),
        "support_score": round(support_score, 4),
        "trend_score": round(trend_score, 4),
        "trend_note": trend_note,
        "width_score": round(width_score, 4),
        "base_score": round(base_score, 4),
    }


def check_box_breakout(df):
    if df is None or df.empty or len(df) < max(BOX_WINDOWS) + 2:
        return {"matched": False, "reason": "bars_not_enough"}

    trend_score, trend_note = calc_trend_score(df)

    valid_results = []
    for w in BOX_WINDOWS:
        result = check_one_window(df, w, trend_score, trend_note)
        if result is not None:
            valid_results.append(result)

    if not valid_results:
        return {"matched": False, "reason": "no_valid_box_breakout"}

    best = max(valid_results, key=lambda x: x["base_score"])
    best_window = int(best["window"])
    valid_windows = sorted(int(x["window"]) for x in valid_results)
    valid_window_count = len(valid_windows)
    neighbor_windows = [w for w in valid_windows if abs(w - best_window) <= 10]

    window_support_score = score_window_support(valid_window_count)
    final_score = best["base_score"] + window_support_score
    signal_grade = classify_signal(final_score)

    best["valid_windows"] = "|".join(map(str, valid_windows))
    best["valid_window_count"] = valid_window_count
    best["neighbor_windows"] = "|".join(map(str, neighbor_windows))
    best["neighbor_valid_count"] = len(neighbor_windows)
    best["window_support_score"] = round(window_support_score, 4)
    best["final_score"] = round(final_score, 4)
    best["signal_grade"] = signal_grade
    best["pattern_note"] = build_pattern_note(best)

    return best


# =========================
# 单股多日期评估
# =========================
def evaluate_one_stock_multi_dates(pro, ts_code: str, ticker: str, name: str, target_dates, start_date, end_date):
    matched_list = []
    debug_list = []
    failed_list = []

    try:
        raw = fetch_tushare_daily_with_retry(pro, ts_code, start_date, end_date, max_retry=3)
    except Exception as e:
        failed_list.append({
            "ticker": ticker,
            "ts_code": ts_code,
            "name": name,
            "target_date": "",
            "error": f"daily_fetch_exception: {type(e).__name__}: {e}",
        })
        return matched_list, debug_list, failed_list

    try:
        df_full = standardize_tushare_daily(raw)
    except Exception as e:
        failed_list.append({
            "ticker": ticker,
            "ts_code": ts_code,
            "name": name,
            "target_date": "",
            "error": f"standardize_exception: {type(e).__name__}: {e}",
        })
        return matched_list, debug_list, failed_list

    if df_full.empty:
        failed_list.append({
            "ticker": ticker,
            "ts_code": ts_code,
            "name": name,
            "target_date": "",
            "error": "no_data",
        })
        return matched_list, debug_list, failed_list

    for target_date in target_dates:
        try:
            target_ts = pd.to_datetime(target_date, format="%Y%m%d")
            df = df_full[df_full["date"] <= target_ts].copy()
            df = df.sort_values("date").reset_index(drop=True)

            if df.empty:
                debug_list.append({
                    "ticker": ticker,
                    "ts_code": ts_code,
                    "name": name,
                    "target_date": target_date,
                    "actual_trade_date": "",
                    "error": "no_data_before_target_date",
                    "reason_list": "no_data_before_target_date",
                })
                continue

            actual_trade_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
            target_date_fmt = target_ts.strftime("%Y-%m-%d")

            # 严格要求 signal_date / actual_trade_date 与 target_date 是同一天。
            # 如果 target_date 不是交易日，或者个股当天停牌/无数据，就不使用前一个交易日代替。
            if actual_trade_date != target_date_fmt:
                debug_list.append({
                    "ticker": ticker,
                    "ts_code": ts_code,
                    "name": name,
                    "target_date": target_date,
                    "actual_trade_date": actual_trade_date,
                    "signal_date": "",
                    "error": "target_date_no_exact_trade_data",
                    "reason_list": f"target_date={target_date_fmt}, actual_trade_date={actual_trade_date}",
                })
                continue

            if len(df) < max(BOX_WINDOWS) + 2:
                debug_list.append({
                    "ticker": ticker,
                    "ts_code": ts_code,
                    "name": name,
                    "target_date": target_date,
                    "actual_trade_date": actual_trade_date,
                    "error": "not_enough_bars",
                    "reason_list": "not_enough_bars",
                })
                continue

            chk = check_box_breakout(df)

            if chk.get("matched", False):
                matched_list.append({
                    "ticker": ticker,
                    "ts_code": ts_code,
                    "name": name,
                    "target_date": target_date,
                    "signal_date": chk["date"],
                    "actual_trade_date": chk["date"],
                    **{k: v for k, v in chk.items() if k not in ["matched", "date"]},
                })
            else:
                debug_list.append({
                    "ticker": ticker,
                    "ts_code": ts_code,
                    "name": name,
                    "target_date": target_date,
                    "actual_trade_date": actual_trade_date,
                    "error": chk.get("reason", "not_matched"),
                    "reason_list": chk.get("reason", "not_matched"),
                })

            if SLEEP_SEC > 0:
                time.sleep(SLEEP_SEC)

        except Exception as e:
            failed_list.append({
                "ticker": ticker,
                "ts_code": ts_code,
                "name": name,
                "target_date": target_date,
                "error": f"check_exception: {type(e).__name__}: {e}",
            })

    return matched_list, debug_list, failed_list


# =========================
# 输出路径
# =========================
def make_output_paths(end_date: str):
    return {
        "output": os.path.join(OUTPUT_DIR, f"box_breakout_candidates_{end_date}.csv"),
        "debug": os.path.join(OUTPUT_DIR, f"box_breakout_debug_rejected_{end_date}.csv"),
        "failed": os.path.join(OUTPUT_DIR, f"box_breakout_failed_fetch_{end_date}.csv"),
        "filtered": os.path.join(OUTPUT_DIR, f"box_breakout_filtered_out_{end_date}.csv"),
    }


# =========================
# 主程序
# =========================
def main():
    if not TUSHARE_TOKEN:
        raise ValueError("未检测到环境变量 TUSHARE_TOKEN，请在 GitHub Secrets 中配置。")

    target_dates = get_target_dates()
    end_date = max(target_dates)
    output_paths = make_output_paths(end_date)

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    start_time = time.perf_counter()

    print("1) 读取仓库 data 目录股票列表，并剔除 ST / 80亿以下 ...")
    universe, filtered_out = load_universe_from_csv()

    print("原始待扫描股票数:", len(universe))
    print("过滤掉数量(ST/80亿以下):", len(filtered_out))
    print("TARGET_DATES:", target_dates)
    print("START_DATE:", START_DATE)
    print("END_DATE:", end_date)
    print(f"扫描窗口: {BOX_WINDOWS[0]}~{BOX_WINDOWS[-1]}，步长 {BOX_STEP}，共 {len(BOX_WINDOWS)} 个窗口")
    print("识别模式: 宽松候选 + 非淘汰式评分分级")
    print("箱体宽度: 不硬淘汰，只做分类和评分")
    print("箱顶偏差: 不硬淘汰，只做箱顶质量评分")

    print("\n2) 开始 Tushare 快速版箱体突破扫描...")

    matched = []
    debug_rejected = []
    failed_fetch = []
    error_counter = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(
                evaluate_one_stock_multi_dates,
                pro,
                row["ts_code"],
                row["ticker"],
                row["name"],
                target_dates,
                START_DATE,
                end_date,
            ): row["ts_code"]
            for _, row in universe.iterrows()
        }

        total = len(future_map)
        for i, future in enumerate(as_completed(future_map), 1):
            matched_list, debug_list, failed_list = future.result()

            matched.extend(matched_list)
            debug_rejected.extend(debug_list)
            failed_fetch.extend(failed_list)

            for item in debug_list:
                err = item.get("error", "unknown")
                error_counter[err] = error_counter.get(err, 0) + 1

            for item in failed_list:
                err = item.get("error", "unknown")
                error_counter[err] = error_counter.get(err, 0) + 1

            if i % 100 == 0 or i == total:
                elapsed_now = time.perf_counter() - start_time
                avg_per_stock = elapsed_now / i if i else 0
                est_total = avg_per_stock * total
                remain = max(0, est_total - elapsed_now)

                rh = int(remain // 3600)
                rm = int((remain % 3600) // 60)
                rs = remain % 60

                print(
                    f"进度: {i}/{total} | 命中: {len(matched)} | 调试未过: {len(debug_rejected)} | 抓取失败: {len(failed_fetch)} "
                    f"| 预计剩余: {rh}小时 {rm}分钟 {rs:.1f}秒"
                )

    matched_df = pd.DataFrame(matched)
    debug_df = pd.DataFrame(debug_rejected)
    failed_df = pd.DataFrame(failed_fetch)
    filtered_df = filtered_out.copy()

    if not matched_df.empty:
        sort_cols = [c for c in [
            "target_date", "signal_grade", "final_score", "top_score", "breakout_score", "valid_window_count"
        ] if c in matched_df.columns]
        ascending_flags = []
        for c in sort_cols:
            ascending_flags.append(True if c in ["target_date", "signal_grade"] else False)
        matched_df = matched_df.sort_values(by=sort_cols, ascending=ascending_flags).reset_index(drop=True)

    matched_df.to_csv(output_paths["output"], index=False, encoding="utf-8-sig")
    debug_df.to_csv(output_paths["debug"], index=False, encoding="utf-8-sig")
    failed_df.to_csv(output_paths["failed"], index=False, encoding="utf-8-sig")
    filtered_df.to_csv(output_paths["filtered"], index=False, encoding="utf-8-sig")

    print("\n命中结果已保存:", output_paths["output"])
    print("调试未通过已保存:", output_paths["debug"])
    print("抓取失败已保存:", output_paths["failed"])
    print("已过滤股票已保存:", output_paths["filtered"])

    print("\n扫描完成")
    print("命中数量:", len(matched_df))
    print("调试未通过数量:", len(debug_df))
    print("抓取失败数量:", len(failed_df))
    print("过滤掉数量(ST/80亿以下):", len(filtered_df))

    if not matched_df.empty and "signal_grade" in matched_df.columns:
        print("\n信号等级统计：")
        print(matched_df["signal_grade"].value_counts())

    if not matched_df.empty and "box_top_quality" in matched_df.columns:
        print("\n箱顶质量统计：")
        print(matched_df["box_top_quality"].value_counts())

    if not matched_df.empty and "box_width_type" in matched_df.columns:
        print("\n箱体宽度类型统计：")
        print(matched_df["box_width_type"].value_counts())

    if not matched_df.empty and "target_date" in matched_df.columns:
        print("\n按日期统计命中数量：")
        print(matched_df.groupby("target_date").size())

    print("\n失败原因统计：")
    for k, v in sorted(error_counter.items(), key=lambda x: -x[1]):
        print(k, v)

    if not matched_df.empty:
        print("\n命中前20条：")
        cols = [
            "ticker", "name", "ts_code", "target_date", "signal_date", "actual_trade_date", "signal_grade", "final_score",
            "window", "valid_windows", "valid_window_count", "box_top_quality", "box_width_type",
            "box_top_price", "box_bottom_price", "box_width_pct", "inbox_ratio_pct",
            "touch_upper_segments", "touch_lower_segments",
            "box_top_touch_dates", "box_top_touch_height_spread_pct", "box_top_touch_close_spread_pct",
            "box_bottom_touch_dates", "box_bottom_touch_height_spread_pct",
            "breakout_pct_vs_top", "top_score", "breakout_score", "stability_score", "support_score", "trend_score",
            "last_close", "pattern_note"
        ]
        cols = [c for c in cols if c in matched_df.columns]
        print(matched_df[cols].head(20).to_string(index=False))

    if not debug_df.empty:
        print("\n调试未通过前20条：")
        cols = ["ticker", "name", "ts_code", "target_date", "actual_trade_date", "error", "reason_list"]
        cols = [c for c in cols if c in debug_df.columns]
        print(debug_df[cols].head(20).to_string(index=False))

    if not failed_df.empty:
        print("\n抓取失败前20条：")
        cols = ["ticker", "name", "ts_code", "target_date", "error"]
        cols = [c for c in cols if c in failed_df.columns]
        print(failed_df[cols].head(20).to_string(index=False))

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print(f"\n总耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    if len(universe) > 0:
        print(f"平均每只股票耗时: {elapsed / len(universe):.2f} 秒")


if __name__ == "__main__":
    main()
