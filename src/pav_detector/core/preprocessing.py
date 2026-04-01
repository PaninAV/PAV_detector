from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def prepare_feature_frame(df: pd.DataFrame, feature_order: List[str]) -> pd.DataFrame:
    if not feature_order:
        numeric_df = df.select_dtypes(include=["number"]).copy()
        return numeric_df.fillna(0.0)

    work = df.copy()
    for feature in feature_order:
        if feature not in work.columns:
            work[feature] = 0.0
    work = work[feature_order]

    for column in work.columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    return work.replace([np.inf, -np.inf], np.nan).fillna(0.0)
