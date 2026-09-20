"""COT parsing: both CFTC schemas, the staleness trap, and unmapped symbols.

parse_cot_rows is pure, so none of this hits the network.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fable_bot.intel.cot import (COMMODITY, FINANCIAL, STALE_REPORT_DAYS, NoCotMarket,
                                 parse_cot_rows)

NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def commodity_row(date_str, mm_long, mm_short, oi=400_000):
    return {
        "market_and_exchange_names": "GOLD - COMMODITY EXCHANGE INC.",
        "report_date_as_yyyy_mm_dd": date_str,
        "open_interest_all": str(oi),
        "prod_merc_positions_long_all": "50000",
        "prod_merc_positions_short_all": "78000",
        "m_money_positions_long_all": str(mm_long),
        "m_money_positions_short_all": str(mm_short),
        "swap_positions_long_all": "20000",
        "swap__positions_short_all": "30000",
    }


def financial_row(date_str, lev_long, lev_short, oi=2_400_000):
    return {
        "market_and_exchange_names": "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
        "report_date_as_yyyy_mm_dd": date_str,
        "open_interest_all": str(oi),
        "dealer_positions_long_all": "120000",
        "dealer_positions_short_all": "822938",
        "lev_money_positions_long_all": str(lev_long),
        "lev_money_positions_short_all": str(lev_short),
        "asset_mgr_positions_long_all": "900000",
        "asset_mgr_positions_short_all": "100000",
    }


def week(n):
    return (NOW - timedelta(days=5 + 7 * n)).date().isoformat()


# ── empty / unmapped ──────────────────────────────────────────────────────

def test_empty_rows_return_none():
    assert parse_cot_rows([]) is None


def test_unmapped_symbol_raises_no_cot_market():
    from fable_bot.intel.cot import fetch_cot
    with pytest.raises(NoCotMarket):
        fetch_cot("OANDA:EURGBP")


# ── disaggregated (commodity) schema ──────────────────────────────────────

def test_commodity_schema_parses_managed_money_and_hedgers():
    report = parse_cot_rows([commodity_row(week(0), 180_000, 47_000)],
                            now=NOW, report_type=COMMODITY)
    assert report.managed_money_net == pytest.approx(133_000)
    assert report.producer_net == pytest.approx(-28_000)
    assert report.spec_label == "managed money"
    assert report.hedger_label == "commercial hedgers"


def test_net_as_pct_of_open_interest():
    report = parse_cot_rows([commodity_row(week(0), 150_000, 50_000, oi=400_000)],
                            now=NOW, report_type=COMMODITY)
    assert report.managed_money_net_pct_oi == pytest.approx(25.0)


# ── TFF (financial) schema ────────────────────────────────────────────────

def test_financial_schema_uses_dealer_and_leveraged_fund_columns():
    report = parse_cot_rows([financial_row(week(0), 200_000, 493_143)],
                            now=NOW, report_type=FINANCIAL)
    assert report.managed_money_net == pytest.approx(-293_143)
    assert report.producer_net == pytest.approx(-702_938)
    assert report.spec_label == "leveraged funds"
    assert "banks" in report.hedger_label


def test_the_two_schemas_do_not_cross_contaminate():
    """Commodity columns must read zero out of a financial row, not silently blend."""
    report = parse_cot_rows([financial_row(week(0), 10, 20)], now=NOW, report_type=COMMODITY)
    assert report.managed_money_net == 0.0


# ── percentile / crowding ─────────────────────────────────────────────────

def test_percentile_ranks_the_newest_week_against_history():
    rows = [commodity_row(week(n), 50_000 + n * 0, 0) for n in range(1, 30)]
    rows.insert(0, commodity_row(week(0), 999_999, 0))  # newest, an extreme
    report = parse_cot_rows(rows, now=NOW, report_type=COMMODITY)
    assert report.managed_money_net_percentile == pytest.approx(96.7, abs=0.5)
    assert report.crowding == "extreme_long"


def test_percentile_is_none_without_enough_history():
    rows = [commodity_row(week(n), 100_000, 10_000) for n in range(3)]
    report = parse_cot_rows(rows, now=NOW, report_type=COMMODITY)
    assert report.managed_money_net_percentile is None
    assert report.crowding == "unknown"


def test_low_percentile_reads_as_extreme_short():
    rows = [commodity_row(week(n), 500_000, 0) for n in range(1, 20)]
    rows.insert(0, commodity_row(week(0), 0, 500_000))
    report = parse_cot_rows(rows, now=NOW, report_type=COMMODITY)
    assert report.crowding == "extreme_short"


# ── staleness ─────────────────────────────────────────────────────────────

def test_normal_lag_is_not_flagged_stale():
    report = parse_cot_rows([commodity_row(week(0), 100_000, 50_000)],
                            now=NOW, report_type=COMMODITY)
    assert report.days_stale == 5
    assert report.is_stale is False


def test_archived_series_is_flagged_stale():
    """A wrong market name matched rows that stopped updating in 2022.

    The record was well-formed and would otherwise have been served as current
    positioning.
    """
    report = parse_cot_rows([commodity_row("2022-02-01", 100_000, 50_000)],
                            now=NOW, report_type=COMMODITY)
    assert report.days_stale > STALE_REPORT_DAYS
    assert report.is_stale is True
    assert "STALE" in report.summary()


def test_rows_are_ordered_newest_first_regardless_of_input_order():
    rows = [commodity_row(week(3), 1, 0), commodity_row(week(0), 2, 0),
            commodity_row(week(1), 3, 0)]
    report = parse_cot_rows(rows, now=NOW, report_type=COMMODITY)
    assert report.report_date == week(0)


def test_unparseable_date_does_not_raise():
    row = commodity_row("not-a-date", 100_000, 50_000)
    report = parse_cot_rows([row], now=NOW, report_type=COMMODITY)
    assert report is not None
    assert report.days_stale == 0


def test_missing_numeric_fields_read_as_zero_not_crash():
    report = parse_cot_rows([{"market_and_exchange_names": "X",
                              "report_date_as_yyyy_mm_dd": week(0)}],
                            now=NOW, report_type=COMMODITY)
    assert report.managed_money_net == 0.0
    assert report.open_interest == 0.0
