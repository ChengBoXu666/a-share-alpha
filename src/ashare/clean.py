"""
Stage 1 作业 · 把原始日线清洗成一张"研究面板"(panel)

面板 = 一张长表，每行是 (date, code)，后面挂着价格、收益、各种状态标记。
之后所有因子、标签、回测都基于这张表，所以这里错一点，后面全错。

👉 你的任务：实现下面每个函数里的 TODO，然后运行 `uv run pytest -q` 直到全部通过。
   全部用 Polars 表达式写（不要 for 循环逐行处理，1200 万行会慢到无法接受）。
   详细说明见 lessons/01_数据层.md
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

GEM_REFORM_DATE = date(2020, 8, 24)  # 创业板注册制改革：涨跌幅 10% → 20%


def add_board(df: pl.DataFrame) -> pl.DataFrame:
    """新增列 board ∈ {"main", "gem", "star"}
    - 科创板 star: 代码以 sh.688 / sh.689 开头
    - 创业板 gem : 代码以 sz.300 / sz.301 开头
    - 其余为主板 main
    """
    # TODO
    is_star = pl.col("code").str.starts_with("sh.688") | pl.col("code").str.starts_with("sh.689")
    is_gem = pl.col("code").str.starts_with("sz.300") | pl.col("code").str.starts_with("sz.301")
    return df.with_columns(
        board=pl.when(is_star).then(pl.lit("star"))
        .when(is_gem).then(pl.lit("gem"))
        .otherwise(pl.lit("main"))
    )
    raise NotImplementedError


def add_status(df: pl.DataFrame) -> pl.DataFrame:
    """新增两列:
    - is_suspended (bool): tradestatus == 0 或 volume == 0
    - float_mv (float): 流通市值 ≈ close * volume / (turn / 100)
        （turn 是换手率%，= 成交量/流通股本×100，所以 流通股本 = volume / (turn/100)）
        turn <= 0 或为空时 float_mv 为 null
    """
    # TODO
    df.with_columns
    raise NotImplementedError


def add_limit_flags(df: pl.DataFrame) -> pl.DataFrame:
    """需要已有列: code, date, board, is_st, preclose, close, volume
    新增列:
    - limit_pct (float): 涨跌幅限制
        科创板 0.20；创业板 date >= 2020-08-24 为 0.20，之前为 0.10（之前的创业板 ST 为 0.05）；
        主板 0.10，主板 ST 为 0.05
    - up_price / down_price: preclose*(1±limit_pct) 【四舍五入】到分
        ⚠️ 浮点陷阱: 5.35*1.1 = 5.885，交易所算出 5.89，但浮点直接 round 会得到 5.88。
           可靠写法: floor(x * 100 + 0.5 + 1e-6) / 100
    - limit_up (bool):   volume > 0 且 close >= up_price - 1e-6
    - limit_down (bool): volume > 0 且 close <= down_price + 1e-6
    """
    # TODO
    raise NotImplementedError


def add_returns(df: pl.DataFrame) -> pl.DataFrame:
    """新增列:
    - ret: 当日收益 = close / preclose - 1  （preclose 已经除权，所以这就是真实收益）
    - adj_close: 每只股票按日期排序后，(1 + ret) 的累乘 —— 相当于"后复权价格指数"
    要求: 输入行顺序可能是乱的，函数内部要先按 (code, date) 排序；输出也按 (code, date) 排序
    """
    # TODO
    raise NotImplementedError


def add_list_days(df: pl.DataFrame, basic: pl.DataFrame) -> pl.DataFrame:
    """basic 有列 code, ipo_date (pl.Date)
    新增列 list_days: date 与 ipo_date 之间的自然日天数（int）
    """
    # TODO
    raise NotImplementedError


def add_universe(df: pl.DataFrame, min_list_days: int = 180) -> pl.DataFrame:
    """新增列 in_universe (bool): 非 ST、非停牌、上市满 min_list_days 天
    —— 这就是"当天可以被我们选入组合的股票池"
    """
    # TODO
    raise NotImplementedError


def build_panel(raw_dir: str | Path = "data/raw/daily",
                meta_dir: str | Path = "data/raw/meta",
                out_path: str | Path = "data/processed/panel.parquet") -> pl.DataFrame:
    """把上面的函数串起来:
    1. pl.scan_parquet 读所有原始文件 → collect；把 date 转成 pl.Date
    2. 读 stock_basic.parquet，ipo_date 转成 pl.Date
    3. 依次调用 add_board → add_status → add_limit_flags → add_returns → add_list_days → add_universe
    4. 写出 out_path，打印行数、股票数、日期范围、每年平均 in_universe 股票数，做 sanity check
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    build_panel()
