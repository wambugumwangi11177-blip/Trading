from fable_bot.strategies import smc


def test_swing_points_shapes(random_walk_df):
    is_sh, is_sl = smc.swing_points(random_walk_df, window=5)
    assert is_sh.dtype == bool
    assert is_sl.dtype == bool
    assert is_sh.sum() > 0
    assert is_sl.sum() > 0


def test_market_structure_labels_uptrend(uptrend_df):
    structure = smc.market_structure(uptrend_df, window=5)
    assert set(structure.unique()) <= {"bullish", "bearish", "neutral"}
    # a clean uptrend should be labeled bullish for most of the back half
    tail = structure.tail(50)
    assert (tail == "bullish").mean() > 0.5


def test_fair_value_gaps_zones_are_ordered(random_walk_df):
    fvg = smc.fair_value_gaps(random_walk_df)
    bullish = fvg.dropna(subset=["bullish_fvg_low", "bullish_fvg_high"])
    bearish = fvg.dropna(subset=["bearish_fvg_low", "bearish_fvg_high"])
    if len(bullish):
        assert (bullish["bullish_fvg_high"] >= bullish["bullish_fvg_low"]).all()
    if len(bearish):
        assert (bearish["bearish_fvg_high"] >= bearish["bearish_fvg_low"]).all()


def test_confluence_series_are_boolean_and_aligned(random_walk_df):
    bull = smc.bullish_confluence(random_walk_df)
    bear = smc.bearish_confluence(random_walk_df)
    assert len(bull) == len(random_walk_df)
    assert bull.dtype == bool
    assert bear.dtype == bool
