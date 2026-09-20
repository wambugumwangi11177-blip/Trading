from fable_bot.strategies.indicators import atr, donchian_channel, ema, zscore


def test_atr_positive_and_aligned(random_walk_df):
    a = atr(random_walk_df, period=14)
    assert len(a) == len(random_walk_df)
    assert (a.dropna() > 0).all()


def test_zscore_centers_around_zero(mean_reverting_df):
    z = zscore(mean_reverting_df["close"], period=20)
    assert abs(z.dropna().mean()) < 1.0
    assert z.dropna().max() > 1.0
    assert z.dropna().min() < -1.0


def test_donchian_channel_bounds(random_walk_df):
    upper, lower = donchian_channel(random_walk_df, period=20)
    valid = upper.notna() & lower.notna()
    assert (upper[valid] >= lower[valid]).all()
    assert (upper[valid] >= random_walk_df["high"][valid]).all()


def test_ema_smooths_less_lag_than_longer_period(uptrend_df):
    fast = ema(uptrend_df["close"], 10)
    slow = ema(uptrend_df["close"], 50)
    # In a clean uptrend the fast EMA should track price more closely (smaller total lag)
    err_fast = (uptrend_df["close"] - fast).abs().tail(50).mean()
    err_slow = (uptrend_df["close"] - slow).abs().tail(50).mean()
    assert err_fast < err_slow
