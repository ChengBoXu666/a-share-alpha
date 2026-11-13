"""Stage 1 验收测试。运行: uv run pytest -q"""
from datetime import date

import polars as pl
import pytest

from ashare.clean import (add_board, add_limit_flags, add_list_days, add_returns,
                          add_status, add_universe)


def test_add_board():
    df = pl.DataFrame({"code": ["sh.600000", "sz.000001", "sz.002594", "sz.300750",
                                "sz.301001", "sh.688981", "sh.689009"]})
    assert add_board(df)["board"].to_list() == ["main", "main", "main", "gem", "gem", "star", "star"]


def test_add_status():
    df = pl.DataFrame({
        "tradestatus": [1, 0, 1, 1],
        "volume": [1000.0, 0.0, 0.0, 2000.0],
        "close": [10.0, 10.0, 10.0, 5.0],
        "turn": [1.0, 0.0, 0.0, None],
    })
    out = add_status(df)
    assert out["is_suspended"].to_list() == [False, True, True, False]
    # 10 * 1000 / (1/100) = 1_000_000
    assert out["float_mv"][0] == pytest.approx(1_000_000.0)
    assert out["float_mv"][1] is None and out["float_mv"][3] is None


CASES = [
    # code,       date,             is_st, preclose, close, limit_up, limit_down, 说明
    ("sh.600000", date(2024, 1, 2), 0, 10.00, 11.00, True,  False),  # 主板涨停
    ("sh.600001", date(2024, 1, 2), 0, 10.00, 10.99, False, False),  # 差一分没涨停
    ("sh.600002", date(2024, 1, 2), 0, 10.00, 9.00,  False, True),   # 主板跌停
    ("sh.600003", date(2024, 1, 2), 1, 10.00, 10.50, True,  False),  # 主板 ST 5%
    ("sz.300001", date(2020, 8, 21), 0, 10.00, 11.00, True, False),  # 改革前创业板 10%
    ("sz.300002", date(2024, 1, 2), 0, 10.00, 11.00, False, False),  # 改革后创业板 20%，+10% 不算涨停
    ("sz.300003", date(2024, 1, 2), 0, 10.00, 12.00, True,  False),
    ("sh.688001", date(2024, 1, 2), 0, 10.00, 8.00,  False, True),   # 科创板 20% 跌停
    ("sh.600004", date(2024, 1, 2), 0, 5.35,  5.88,  False, False),  # 浮点陷阱：涨停价是 5.89 不是 5.88
    ("sh.600005", date(2024, 1, 2), 0, 5.35,  5.89,  True,  False),
]


def test_add_limit_flags():
    df = pl.DataFrame(
        [c[:5] for c in CASES],
        schema=["code", "date", "is_st", "preclose", "close"], orient="row",
    ).with_columns(volume=pl.lit(1000.0))
    out = add_limit_flags(add_board(df))
    assert out["limit_up"].to_list() == [c[5] for c in CASES]
    assert out["limit_down"].to_list() == [c[6] for c in CASES]
    assert out.filter(pl.col("code") == "sh.600004")["up_price"][0] == pytest.approx(5.89)


def test_add_returns_handles_split_and_order():
    # A 股 1/3 发生 10 送 10：前收从 11 被交易所调成 5.5
    df = pl.DataFrame({
        "code": ["A", "B", "A", "A"],
        "date": [date(2024, 1, 3), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2)],
        "preclose": [5.5, 20.0, 10.0, 11.0],
        "close": [6.05, 19.0, 11.0, 11.0],
    })
    out = add_returns(df)
    a = out.filter(pl.col("code") == "A")
    assert a["date"].to_list() == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    assert a["ret"].to_list() == pytest.approx([0.1, 0.0, 0.1])
    assert a["adj_close"].to_list() == pytest.approx([1.1, 1.1, 1.21])
    assert out.filter(pl.col("code") == "B")["ret"][0] == pytest.approx(-0.05)
    assert out["code"].to_list() == ["A", "A", "A", "B"]


def test_add_list_days_and_universe():
    basic = pl.DataFrame({"code": ["A", "B"], "ipo_date": [date(2023, 1, 1), date(2024, 1, 1)]})
    df = pl.DataFrame({
        "id": [0, 1, 2, 3],
        "code": ["A", "A", "B", "A"],
        "date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2), date(2023, 3, 1)],
        "is_st": [0, 1, 0, 0],
        "is_suspended": [False, False, False, False],
    })
    out = add_universe(add_list_days(df, basic)).sort("id")  # join 可能打乱行序
    assert out["list_days"].to_list() == [366, 366, 1, 59]
    assert out["in_universe"].to_list() == [True, False, False, False]
