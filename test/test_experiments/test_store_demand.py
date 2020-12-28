import numpy as np
import pandas as pd

from node.experiments.store_demand.features import load_data, add_features


def test_forward_looking_features():
    df = load_data()
    full_df = df.iloc[:20000, :].copy()
    thinned_df = df.iloc[:5000, :].copy()

    full_df_feats = add_features(full_df).iloc[:5000, :]
    thinned_df_feats = add_features(thinned_df)
    pd.testing.assert_frame_equal(full_df_feats, thinned_df_feats)
