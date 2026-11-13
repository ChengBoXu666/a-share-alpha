"""
Stage 1 · 下载 A 股日线原始数据（BaoStock：免费、无需注册、无需 token）

用法（在项目根目录执行）:
    uv run python scripts/01_download.py --limit 20     # 先试跑 20 只，确认没问题
    uv run python scripts/01_download.py                # 全量，约 5000+ 只，1~3 小时，可断点续传

设计要点（面试会问，务必理解）:
1. 股票列表包含【已退市】股票 —— 只用今天还活着的股票做回测 = 幸存者偏差，回测会虚高。
2. 下载【不复权】价格 + preclose（交易所给的"除权后前收盘价"）:
   - 收益率 ret = close / preclose - 1 天然已经处理了分红送转，不需要复权因子；
   - 涨跌停判断必须用不复权的真实价格（涨停价 = 前收 × 1.1 四舍五入到分）。
3. 每只股票存一个 parquet，已存在就跳过 → 断点续传，中途断网重跑即可。
"""
from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import baostock as bs
import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("data/raw/daily")
META_DIR = Path("data/raw/meta")

# 日线字段: 不复权价格、成交、换手率(%)、交易状态(1正常/0停牌)、是否ST、估值
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST,peTTM,pbMRQ"
RENAME = {"pctChg": "pct_chg", "isST": "is_st", "peTTM": "pe_ttm", "pbMRQ": "pb_mrq"}
FLOAT_COLS = ["open", "high", "low", "close", "preclose", "volume", "amount",
              "turn", "pct_chg", "pe_ttm", "pb_mrq"]
INT_COLS = ["tradestatus", "is_st"]


def rs_to_df(rs) -> pd.DataFrame:
    """把 baostock 的 ResultData 转成 DataFrame（baostock 返回的全是字符串）。"""
    if rs.error_code != "0":
        raise RuntimeError(f"baostock error {rs.error_code}: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def download_meta(start: str, end: str) -> pd.DataFrame:
    META_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 全部证券基本信息（含已退市）。type: 1股票 2指数 3其它; status: 1上市 0退市
    basic = rs_to_df(bs.query_stock_basic())
    basic = basic.rename(columns={"ipoDate": "ipo_date", "outDate": "out_date"})
    basic = basic[(basic["type"] == "1") & (~basic["code"].str.startswith("bj."))].copy()
    for c in ["ipo_date", "out_date"]:
        basic[c] = pd.to_datetime(basic[c], errors="coerce")  # 未退市的 out_date 为空 → NaT
    basic.to_parquet(META_DIR / "stock_basic.parquet", index=False)

    # 2) 行业分类（证监会行业）。注意：这是【当前】行业，用于历史有轻微未来信息，README 里要写明这个局限
    ind = rs_to_df(bs.query_stock_industry())
    ind.to_parquet(META_DIR / "industry.parquet", index=False)

    # 3) 交易日历
    cal = rs_to_df(bs.query_trade_dates(start_date=start, end_date=end))
    cal["calendar_date"] = pd.to_datetime(cal["calendar_date"])
    cal["is_trading_day"] = cal["is_trading_day"].astype(int)
    cal.to_parquet(META_DIR / "trade_calendar.parquet", index=False)

    print(f"股票 {len(basic)} 只（其中已退市 {int((basic['status'] == '0').sum())} 只），行业 {len(ind)} 条")
    return basic


def download_one(code: str, start: str, end: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(code, FIELDS, start_date=start, end_date=end,
                                      frequency="d", adjustflag="3")  # 3 = 不复权
    df = rs_to_df(rs).rename(columns=RENAME)
    if df.empty:
        return df
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # 空字符串 → NaN
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01", help="留出约 1 年给因子计算的回看窗口")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--limit", type=int, default=None, help="只下载前 N 只（试跑用）")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    lg = bs.login()
    if lg.error_code != "0":
        raise SystemExit(f"登录失败: {lg.error_msg}")

    try:
        basic = download_meta(args.start, args.end)
        start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
        # 只保留和下载区间有交集的股票：上市早于 end，且（未退市 或 退市晚于 start）
        alive = basic[(basic["ipo_date"] <= end_ts) &
                      (basic["out_date"].isna() | (basic["out_date"] >= start_ts))]
        codes = alive["code"].tolist()[: args.limit]

        failed = []
        for code in tqdm(codes, desc="下载日线"):
            out = RAW_DIR / f"{code}.parquet"
            if out.exists():
                continue  # 断点续传
            for attempt in range(3):
                try:
                    df = download_one(code, args.start, args.end)
                    df.to_parquet(out, index=False)  # 空表也写，标记"已处理"
                    break
                except Exception as e:  # 网络抖动：重新登录后重试
                    if attempt == 2:
                        failed.append((code, repr(e)))
                    time.sleep(2)
                    bs.logout()
                    bs.login()
        if failed:
            print(f"失败 {len(failed)} 只，重跑脚本即可补齐: {failed[:5]} ...")
        print("完成 ✅")
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
