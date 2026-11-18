"""
Stage 1 · 下载 A 股日线原始数据（BaoStock：免费、无需注册、无需 token）

用法（在项目根目录执行）:
    uv run python scripts/01_download.py --limit 20     # 先试跑 20 只，确认没问题
    uv run python scripts/01_download.py                # 全量，约 5000+ 只，耗时取决于网络（几小时），可断点续传

设计要点（面试会问，务必理解）:
1. 股票列表包含【已退市】股票 —— 只用今天还活着的股票做回测 = 幸存者偏差，回测会虚高。
2. 下载【不复权】价格 + preclose（交易所给的"除权后前收盘价"）:
   - 收益率 ret = close / preclose - 1 天然已经处理了分红送转，不需要复权因子；
   - 涨跌停判断必须用不复权的真实价格（涨停价 = 前收 × 1.1 四舍五入到分）。
3. 每只股票存一个 parquet，已存在就跳过 → 断点续传，中途断网重跑即可。
4. 多进程并行下载（--workers，默认 4）；网络请求设 60 秒超时，卡住会自动重试而不是一直挂着。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import socket
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
        # 空字符串 → NaN；必须显式转 float64：否则某只股票 volume 全是整数时会被推断成 int64，
        # 另一只有空值的是 float64，各文件列类型不一致，合并读取时报 SchemaError
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _init_worker() -> None:
    """每个下载进程各自登录一次。超时设置让"卡死"变成"报错后重试"。"""
    socket.setdefaulttimeout(60)
    bs.login()


def _fetch(task: tuple[str, str, str]) -> tuple[str, str]:
    """下载并保存一只股票。返回 (code, 状态)：ok / empty / 错误信息"""
    code, start, end = task
    out = RAW_DIR / f"{code}.parquet"
    err = ""
    for _ in range(3):
        try:
            df = download_one(code, start, end)
            if df.empty:
                return code, "empty"
            tmp = out.with_suffix(".tmp")  # 先写临时文件再改名：不会留下写了一半的 parquet
            df.to_parquet(tmp, index=False)
            tmp.replace(out)
            return code, "ok"
        except Exception as e:  # 网络抖动 / 超时：重新登录后重试
            err = repr(e)
            time.sleep(2)
            try:
                bs.logout()
            except Exception:
                pass
            bs.login()
    return code, err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01", help="留出约 1 年给因子计算的回看窗口")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--limit", type=int, default=None, help="只下载前 N 只（试跑用）")
    ap.add_argument("--workers", type=int, default=4, help="并行下载的进程数")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    socket.setdefaulttimeout(60)
    lg = bs.login()
    if lg.error_code != "0":
        raise SystemExit(f"登录失败: {lg.error_msg}")
    try:
        basic = download_meta(args.start, args.end)
    finally:
        bs.logout()

    start_ts, end_ts = pd.Timestamp(args.start), pd.Timestamp(args.end)
    # 只保留和下载区间有交集的股票：上市早于 end，且（未退市 或 退市晚于 start）
    alive = basic[(basic["ipo_date"] <= end_ts) &
                  (basic["out_date"].isna() | (basic["out_date"] >= start_ts))]
    codes = alive["code"].tolist()[: args.limit]

    no_data_file = META_DIR / "no_data_codes.txt"
    no_data = set(no_data_file.read_text().split()) if no_data_file.exists() else set()
    # 断点续传：已有文件的、已知没数据的都跳过
    todo = [c for c in codes if c not in no_data and not (RAW_DIR / f"{c}.parquet").exists()]
    print(f"共 {len(codes)} 只，已完成 {len(codes) - len(todo)} 只，本次下载 {len(todo)} 只，{args.workers} 个进程")

    failed = []
    tasks = [(c, args.start, args.end) for c in todo]
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for code, status in tqdm(pool.imap_unordered(_fetch, tasks), total=len(tasks), desc="下载日线"):
            if status == "empty":
                with no_data_file.open("a") as f:
                    f.write(code + "\n")
            elif status != "ok":
                failed.append((code, status))

    if failed:
        print(f"失败 {len(failed)} 只，重跑脚本即可补齐: {failed[:5]} ...")
    else:
        print("完成 ✅")


if __name__ == "__main__":
    main()
