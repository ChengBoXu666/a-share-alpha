"""
Stage 2 · 因子库

设计：每个因子是一个函数，返回一个 Polars 表达式 (pl.Expr)，描述"对【一只股票】的时间序列怎么算"。
- 写因子时只需要考虑单只股票，不用管分组；
- compute_factors 会统一按 (code, date) 排序，并给每个表达式加上 .over("code")，
  保证每只股票各算各的，前一只股票的数据不会漏进下一只。

铁律（面试必问）：t 日的因子值只能用 t 日收盘及以前的数据。
  - 只允许 shift(正数)（看过去），禁止 shift(负数)（看未来）
  - 滚动窗口 rolling_xxx 默认只看当前行及之前的行，是安全的
  tests/test_factors.py 里有一个"篡改未来数据"的测试，专门抓这类错误。

收益类因子用 adj_close（复权价格指数）或 ret，不要用原始 close（会被送转的假跳空污染）。

👉 你的任务：实现每个函数里的 TODO，运行 `uv run pytest -q tests/test_factors.py` 直到全部通过。
   详细说明见 lessons/02_因子库.md
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


# ============ 动量 / 反转 ============

def mom(n: int, skip: int = 0) -> pl.Expr:
    """过去收益率：从 t-n 日收盘到 t-skip 日收盘的收益
        adj_close[t-skip] / adj_close[t-n] - 1
    - mom(5): 过去 5 日收益（A 股短期通常表现为【反转】）
    - mom(250, skip=20): 过去一年、但跳过最近一个月的收益（经典中期动量写法）
    """
    # TODO
    raise NotImplementedError


# ============ 波动率 ============

def vol(n: int) -> pl.Expr:
    """过去 n 日日收益率 ret 的标准差（样本标准差，ddof=1，Polars 默认就是）"""
    # TODO
    raise NotImplementedError


def max_ret(n: int) -> pl.Expr:
    """过去 n 日里最大的单日收益（"彩票型"股票：最近有过暴涨的股票，未来往往跑输）"""
    # TODO
    raise NotImplementedError


# ============ 流动性 / 换手 ============

def turn_mean(n: int) -> pl.Expr:
    """过去 n 日平均换手率 turn"""
    # TODO
    raise NotImplementedError


def abn_turn(short: int, long: int) -> pl.Expr:
    """异常换手：近期平均换手 / 长期平均换手 = turn_mean(short) / turn_mean(long)
    （提示：可以直接调用上面写好的 turn_mean）"""
    # TODO
    raise NotImplementedError


def amihud(n: int) -> pl.Expr:
    """Amihud 非流动性：过去 n 日 |ret| / amount 的平均值
    —— 每成交 1 元能把价格推动多少。越大越不流动。
    amount <= 0（停牌）时当日值为 null，不能除以 0
    """
    # TODO
    raise NotImplementedError


# ============ 量价关系 ============

def pv_corr(n: int) -> pl.Expr:
    """过去 n 日 adj_close 与 volume 的相关系数（价量同涨同跌 vs 背离）
    提示：pl.rolling_corr(表达式a, 表达式b, window_size=n)
    """
    # TODO
    raise NotImplementedError


# ============ 规模 / 估值（截面因子：不需要回看窗口） ============

def log_size() -> pl.Expr:
    """流通市值取自然对数 ln(float_mv)。取对数是因为市值跨度极大（几亿到几万亿）"""
    # TODO
    raise NotImplementedError


def ep() -> pl.Expr:
    """盈利收益率 E/P = 1 / pe_ttm（市盈率的倒数）
    pe_ttm == 0 时为 null。注意：亏损公司 pe 为负，E/P 也为负 —— 这是合理的，保留
    （为什么用 1/PE 而不是 PE：PE 在盈利接近 0 时会趋于无穷、正负跳变，倒数是连续单调的）
    """
    # TODO
    raise NotImplementedError


def bp() -> pl.Expr:
    """账面市值比 B/P = 1 / pb_mrq，规则同 ep()"""
    # TODO
    raise NotImplementedError


# ============ 因子注册表 + 批量计算（这部分已写好） ============

def factor_exprs() -> dict[str, pl.Expr]:
    """所有因子的名字 → 表达式。以后加新因子就在这里登记一行。"""
    return {
        "mom_5": mom(5),
        "mom_20": mom(20),
        "mom_12_1": mom(250, skip=20),
        "vol_20": vol(20),
        "max_ret_20": max_ret(20),
        "turn_20": turn_mean(20),
        "abn_turn_5_120": abn_turn(5, 120),
        "amihud_20": amihud(20),
        "pv_corr_20": pv_corr(20),
        "log_size": log_size(),
        "ep": ep(),
        "bp": bp(),
    }


def compute_factors(panel: pl.DataFrame) -> pl.DataFrame:
    """对面板计算全部因子。返回 (date, code, in_universe, 各因子列)"""
    exprs = factor_exprs()
    return (
        panel.sort("code", "date")
        .with_columns([e.over("code").alias(name) for name, e in exprs.items()])
        .select("date", "code", "in_universe", *exprs.keys())
    )


def build_factors(panel_path: str | Path = "data/processed/panel.parquet",
                  out_path: str | Path = "data/processed/factors.parquet") -> pl.DataFrame:
    panel = pl.read_parquet(panel_path)
    fac = compute_factors(panel)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fac.write_parquet(out_path)

    # sanity check：在股票池内，每个因子的缺失比例和分布
    names = list(factor_exprs().keys())
    uni = fac.filter(pl.col("in_universe"))
    print(f"因子表 {fac.height:,} 行；股票池内 {uni.height:,} 行")
    summary = pl.DataFrame({
        "factor": names,
        "null_pct": [round(uni[c].null_count() / uni.height * 100, 2) for c in names],
        "p01": [uni[c].quantile(0.01) for c in names],
        "median": [uni[c].median() for c in names],
        "p99": [uni[c].quantile(0.99) for c in names],
    })
    with pl.Config(tbl_rows=50, float_precision=4):
        print(summary)
    return fac


if __name__ == "__main__":
    build_factors()
