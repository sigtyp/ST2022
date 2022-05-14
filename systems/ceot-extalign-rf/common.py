"""
Data and code common to different modules of the submission.
"""

# todo: mark prosody and other continuous as non categorical (but needs to consider encoding)

# Import Python standard libraries
from typing import *
import random
import hashlib

# Import 3rd-party libraries
import pandas as pd
import numpy as np

import multitiers

# Training methods available and implemented
METHODS = [
    # "ada",
    "dt",
    "knc",
    "lgb",
    "mlp",
    "svc",
    "rf",
]


def set_seeds(seed: Optional[Hashable] = None) -> int:
    """
    Set seeds globally from the user provided one.
    The function takes care of reproducibility and allows using strings and
    floats as seed for `numpy` as well.
    """

    # Set Python generator seed
    random.seed(seed)
    new_seed = random.randint(0, (2**32) - 1)

    # Allows using strings as np seeds, which only takes uint32 or arrays of
    # numbers. As this does not allow accepting `None`, we manually
    # call the function if necessary,
    if seed is None:
        np.random.seed()
    else:
        # Set the np seed
        if isinstance(seed, (str, float)):
            np_seed = np.frombuffer(
                hashlib.sha256(str(seed).encode("utf-8")).digest(), dtype=np.uint32
            )
            np.random.seed(np_seed)
        else:
            np.random.seed(seed)

    return new_seed


def read_mtdf_data(
    filename: str, drop_noninfo: bool = True
) -> Tuple[pd.DataFrame, list]:
    """
    Read data and doculects.
    """

    # Load data and extract doculect names
    mt = pd.read_csv(filename, sep="\t", encoding="utf-8")
    doculects = sorted([col[3:] for col in [col for col in mt.columns if "id_" in col]])

    # Drop columns that offer no prediction or overfit; these should be only kept
    # for testing, debugging, etc.
    if drop_noninfo:
        del mt["cogid"]
        for doculect in doculects:
            del mt[f"id_{doculect}"]

    return mt, doculects


def get_X_y(
    mt: pd.DataFrame,
    doculect: str,
    method: str,
    X_columns: Optional[list] = None,
    encode: bool = False,
):
    # drop entries without the observation in target (missing cognate)
    missing = pd.notna(mt[f"segment_{doculect}"])
    filtered = mt[missing]

    # Get (raw) X and y; if `fillna` is set, replace all NANs by its value
    # before encoding
    y = filtered[f"segment_{doculect}"]
    X = filtered[[col for col in mt.columns if doculect not in col]]

    # encode if requested -- note that this is usually only done when running the classifier
    if encode:
        X_enc = multitiers.IndicatorEncoder()
        X_enc.fit(X.columns)
        X = X_enc.transform(X)

    # As there might be differences in the column names due to the encoding
    # (for example, test data missing values observed in training), we
    # need to make sure that there is a match if column names are provided
    # TODO: rename to avoid confusion between X_columns and X.columns
    if X_columns:
        missing = [feat for feat in X_columns if feat not in X.columns]
        df_missing = pd.DataFrame(
            np.full([len(X.index), len(missing)], fill_value=np.nan),
            index=X.index,
            columns=missing,
        )  # TODO: NAN?
        X = X.join(df_missing)
        X = X[X_columns]  # reorder

    # Mark categorical columns as such
    for col in X.columns:
        col_type = X[col].dtype
        if col_type.name in ["object", "category"]:
            X[col] = X[col].astype("category")

    return X, y
