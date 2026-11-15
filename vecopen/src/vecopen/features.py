"""Phase 6: the 12 state features (SPEC section 4).

Planned API: ``FeatureBuilder.update(bar) -> outputs`` (streaming only, standing rule 2).
The closed list: tf_rank, leg_size, leg_vol, age, confluence, approach_speed, approach_vol,
absorption, trap_wick, delta_div, heat, tod. All are scale-free and every rolling statistic
is shifted by one bar (section 6, construction rule 2).

Stub: nothing is implemented in phase 0.
"""
