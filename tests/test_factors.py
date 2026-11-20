"""Stage 2 验收测试。运行: uv run pytest -q tests/test_factors.py
单个因子: uv run pytest -q tests/test_factors.py -k mom
"""
import math

import numpy as np
import polars as pl
import pytest

from ashare.factors import (abn_turn, amihud, bp, compute_factors, ep, factor_exprs,
                            log_size, max_ret, mom, pv_corr, turn_mean, vol)


def run(expr: pl.Expr, **cols) -> list:
    """在一只股票的小表上计算一个因子表达式，返回结果列表"""
    return pl.DataFrame(cols).select(expr.alias("f"))["f"].to_list()


def close(a: list, b: list) -> bool:
    """逐个比较，None 必须对 None，数值约等于"""
    assert len(a) == len(b), (a, b)
    for x, y in zip(a, b):
        if y is None:
            assert x is None, (a, b)
        else:
            assert x == pytest.approx(y), (a, b)
    return True


# ---------------- 单个因子 ----------------

def test_mom():
    px = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert close(run(mom(2), adj_close=px), [None, None, 2.0, 1.0, 5 / 3 - 1, 6 / 4 - 1])
    # 跳过最近 1 天: adj_close[t-1] / adj_close[t-2] - 1
    assert close(run(mom(2, skip=1), adj_close=px), [None, None, 1.0, 0.5, 4 / 3 - 1, 5 / 4 - 1])


def test_vol():
    assert close(run(vol(3), ret=[0.01, 0.02, 0.03, 0.05]), [None, None, 0.01, 0.0152753])


def test_max_ret():
    assert close(run(max_ret(3), ret=[0.01, 0.05, -0.02, 0.03, -0.01]), [None, None, 0.05, 0.05, 0.03])


def test_turn_mean_and_abn_turn():
    turn = [1.0, 3.0, 5.0, 7.0]
    assert close(run(turn_mean(2), turn=turn), [None, 2.0, 4.0, 6.0])
    # 近 1 日 / 近 2 日均值
    assert close(run(abn_turn(1, 2), turn=turn), [None, 3 / 2, 5 / 4, 7 / 6])


def test_amihud():
    out = run(amihud(2), ret=[0.01, -0.02, 0.03], amount=[100.0, 100.0, 300.0])
    # |ret|/amount = [1e-4, 2e-4, 1e-4]
    assert close(out, [None, 1.5e-4, 1.5e-4])
    # 成交额为 0 时不能出现无穷大
    out0 = run(amihud(1), ret=[0.01, 0.0], amount=[100.0, 0.0])
    assert out0[1] is None or math.isnan(out0[1])


def test_pv_corr():
    out = run(pv_corr(3), adj_close=[1.0, 2.0, 3.0, 4.0], volume=[10.0, 20.0, 30.0, 10.0])
    assert close(out, [None, None, 1.0, -0.5])


def test_log_size():
    assert close(run(log_size(), float_mv=[1e9, None]), [math.log(1e9), None])


def test_ep_bp():
    assert close(run(ep(), pe_ttm=[10.0, -20.0, 0.0, None]), [0.1, -0.05, None, None])
    assert close(run(bp(), pb_mrq=[2.0, 0.0, -1.0]), [0.5, None, -1.0])


# ---------------- 整体性质 ----------------

def _random_stock(n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.02, n)
    return pl.DataFrame({
        "ret": ret,
        "adj_close": np.cumprod(1 + ret),
        "turn": rng.uniform(0.5, 5, n),
        "volume": rng.uniform(1e5, 1e7, n),
        "amount": rng.uniform(1e6, 1e8, n),
        "float_mv": rng.uniform(1e9, 1e11, n),
        "pe_ttm": rng.normal(20, 30, n),
        "pb_mrq": rng.uniform(0.5, 5, n),
    })


@pytest.mark.parametrize("name", list(factor_exprs().keys()))
def test_no_lookahead(name):
    """最重要的测试：篡改第 k 天以后的数据，第 k 天及以前的因子值必须完全不变。
    如果变了，说明这个因子偷看了未来。"""
    expr = factor_exprs()[name]
    k = 300
    df = _random_stock()
    tampered = df.with_columns(
        [pl.when(pl.int_range(pl.len()) > k).then(pl.col(c) * 3.7 + 1.0).otherwise(pl.col(c)).alias(c)
         for c in df.columns]
    )
    a = df.select(expr.alias("f"))["f"].head(k + 1).to_list()
    b = tampered.select(expr.alias("f"))["f"].head(k + 1).to_list()
    assert a == b, f"{name} 用到了未来数据"
    assert any(x is not None for x in a), f"{name} 前 {k} 天全是空值，检查窗口"


def test_compute_factors_no_cross_stock_leak():
    """两只股票的行交错排列：B 的第一天不能用到 A 的数据"""
    a = _random_stock(30, seed=1).with_columns(code=pl.lit("A"))
    b = _random_stock(30, seed=2).with_columns(code=pl.lit("B"))
    panel = (
        pl.concat([a, b])
        .with_columns(date=pl.int_range(pl.len()).mod(30), in_universe=pl.lit(True),
                      _r=pl.Series(np.random.default_rng(0).permutation(60)))
        .sort("_r").drop("_r")  # 打乱行序
    )
    out = compute_factors(panel)
    first_b = out.filter((pl.col("code") == "B") & (pl.col("date") == 0))
    assert first_b["mom_5"][0] is None
    assert out.height == 60
