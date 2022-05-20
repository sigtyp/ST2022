#!/usr/bin/env python3

# For visualization:
# https://www.kaggle.com/code/mlisovyi/beware-of-categorical-features-in-lgbm/notebook
# https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/

"""
Train classifiers using multitiered data.
"""

# todo: try different convergence for NN

# Import Python standard libraries
from pathlib import Path
import argparse
import glob
import logging
from typing import *
import pprint

# Import 3rd-party libraries
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
import joblib
import optuna

# Import other modules
import common
import multitiers

BASE_PATH = Path(__file__).parent


def add_missing_y_values(X_train, y_train, y_test, model: str):
    """
    Adds missing y values to X.
    """

    # if `y_test` is None or empty (does not exist) there is nothing to add; we need to check
    # with list() ad the truth value of a Series (what y_test would be in normal cases) is
    # ambiguous
    if y_test is None or not list(y_test):
        return X_train, y_train

    # Collect missing values and immediately return if there is none
    missing = [y_value for y_value in set(y_test) if y_value not in set(y_train)]
    if not missing:
        return X_train, y_train

    # Cache datatypes
    # Note: seems pandas intended behaviour, even if it seems a bit inconsistent
    orig_types = {col: X_train[col].dtype.name for col in X_train.columns}

    # Cache an empty X_row, with zeros for integer columns (where we cannot attribute nans);
    # note that the missing value might be different depending on the model we are using
    # (basically, if it automatically handles NAs or not)
    if model == "lgb":
        miss_val = np.nan
    else:
        miss_val = 0

    X_row = [
        miss_val if not dtype.name.startswith("int") else 0
        for col, dtype in zip(X_train.columns, X_train.dtypes)
    ]

    # Add all missing values
    next_idx = max(X_train.index)
    for value in missing:
        next_idx = next_idx + 1
        X_train.loc[next_idx] = X_row
        y_train.loc[next_idx] = value

    # Cast back the original cached datatypes
    for col, type_name in orig_types.items():
        X_train[col] = X_train[col].astype(type_name)

    return X_train, y_train


def add_missing_X_columns(X, columns, fill_value=0):
    """
    Add missing columns to a dataframe.

    This function is used to expand the X dataframe when specific values where not
    observed in training, particularly with binary encoding.

    Note that it also takes care of reordering the columns, as expected by some
    prediction methods.
    """
    missing = [feat for feat in columns if feat not in X.columns]
    df_missing = pd.DataFrame(
        np.full([len(X.index), len(missing)], fill_value=fill_value),
        index=X.index,
        columns=missing,
    )
    X = X.join(df_missing)
    X = X[columns]  # reorder

    return X


def build_classifier(mt_train, mt_ref, doculect, method, runargs):
    """
    Build and return a classifier using hyperparameter optimization.
    """

    # Obtain full X and y for training and testing
    X, y = common.get_X_y(mt_train, doculect, method)
    if runargs["use_reference"]:
        X_ref, y_ref = common.get_X_y(mt_ref, doculect, method, list(X.columns))
    else:
        X_ref, y_ref = None, []

    # Given that `y_ref` might contain previously unseen labels, it is
    # better to make sure that all are observed in training
    X, y = add_missing_y_values(X, y, y_ref, method)

    def _objective(trial):
        """
        Internal optune function for performing optimization.
        """

        strat_kf = StratifiedKFold(
            n_splits=runargs["kfolds"], shuffle=True, random_state=runargs["seed"]
        )
        scores = np.empty(runargs["kfolds"])
        for idx, (train_idx, test_idx) in enumerate(strat_kf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Expand with any missing values
            X_train, y_train = add_missing_y_values(X_train, y_train, y_test, method)

            # Build a new classifier at each round; if the classifier requires encoding and
            # NA handling, perform it first; note that we cannot perform this above, as
            # the stratified sampling would fail
            if method not in ["lgb"]:
                # Build encoder
                X_enc = multitiers.IndicatorEncoder()
                X_enc.fit(X_train.columns)

                # Encode training and test data
                X_train = X_enc.transform(X_train)
                X_test = X_enc.transform(X_test)
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)

                # Make sure training and test data have the same set of columns
                all_cols = sorted(set(list(X_train.columns) + list(X_test.columns)))
                X_train = add_missing_X_columns(X_train, all_cols)
                X_test = add_missing_X_columns(X_test, all_cols)

            # todo: join the sklearn methods, and distribute methods in their own functions/files
            if method == "svc":
                _clf = svm.SVC(
                    kernel=trial.suggest_categorical(
                        "kernel", ["linear", "poly", "rbf"]
                    ),
                    degree=trial.suggest_int("degree", 2, 4),
                    decision_function_shape=trial.suggest_categorical(
                        "decision_function_shape", ["ovo", "ovr"]
                    ),
                    random_state=runargs["seed"],
                )
                _clf.fit(X_train, y_train)
            elif method == "dt":
                _clf = tree.DecisionTreeClassifier(
                    min_samples_split=trial.suggest_float(
                        "min_samples_split", 0.0, 0.5
                    ),
                    random_state=runargs["seed"],
                )
                _clf.fit(X_train, y_train)
            elif method == "rf":
                _clf = RandomForestClassifier(
                    min_samples_split=trial.suggest_float(
                        "min_samples_split", 0.0, 0.5
                    ),
                    random_state=runargs["seed"],
                )
                _clf.fit(X_train, y_train)
            elif method == "mlp":
                n_layers = trial.suggest_int("n_layers", 1, 4)
                layers = []
                for i in range(n_layers):
                    layers.append(trial.suggest_int(f"n_units_{i}", 1, 128))

                _clf = MLPClassifier(
                    hidden_layer_sizes=tuple(layers), random_state=runargs["seed"]
                )
                _clf.fit(X_train, y_train)
            elif method == "knc":
                _clf = KNeighborsClassifier(
                    n_neighbors=trial.suggest_int("n_neighbors", 1, 10),
                    weights=trial.suggest_categorical(
                        "weights", ["uniform", "distance"]
                    ),
                )
                _clf.fit(X_train, y_train)
            elif method == "ada":
                _clf = AdaBoostClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 30, 60),
                    learning_rate=trial.suggest_float("learning_rate", 0.5, 1.0),
                    random_state=runargs["seed"],
                )
                _clf.fit(X_train, y_train)
            elif method == "lgb":
                _clf = lgb.LGBMClassifier(
                    application="multiclassova",
                    boosting_type="dart",
                    max_depth=-1,
                    num_class=len(set(list(y) + list(y_ref))),
                    # bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    # feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
                    # learning_rate=trial.suggest_float("learning_rate", 0.5, 1.0),
                    # max_bin=trial.suggest_int("max_bin", 63, 255),
                    min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 30),
                    min_data_per_group=trial.suggest_int("min_data_per_group", 10, 30),
                    # n_estimators=trial.suggest_int("n_estimators", 50, 200),
                    # num_iterations=trial.suggest_int("num_iterations", 100, 200),
                    num_leaves=trial.suggest_int("num_leaves", 15, 63),
                )

                _clf.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=-1,
                )
            else:
                raise ValueError("Unknown method.")

            # Update accuracy tracker
            if runargs["use_reference"]:
                y_pred = _clf.predict(X_ref)
                accuracy = accuracy_score(y_ref, y_pred)
            else:
                y_pred = _clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

            scores[idx] = accuracy

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=runargs["trials"])

    # Retrain with the parameters above, using full data (thus we cannot just
    # take over the best fit); encoding X now that the optimization is over
    if method not in ["lgb"]:
        # Build encoder and encode
        X_enc = multitiers.IndicatorEncoder()
        X_enc.fit(X.columns)
        X = X_enc.transform(X)
        X = X.fillna(0)

    # Build the classifier for returning
    if method == "svc":
        clf = svm.SVC(
            kernel=study.best_trial.params["kernel"],
            degree=study.best_trial.params["degree"],
            decision_function_shape=study.best_trial.params["decision_function_shape"],
            random_state=runargs["seed"],
        )
    elif method == "dt":
        clf = tree.DecisionTreeClassifier(
            min_samples_split=study.best_trial.params["min_samples_split"],
            random_state=runargs["seed"],
        )
    elif method == "rf":
        clf = RandomForestClassifier(
            min_samples_split=study.best_trial.params["min_samples_split"],
            random_state=runargs["seed"],
        )
    elif method == "mlp":
        layers = [
            study.best_trial.params[f"n_units_{i}"]
            for i in range(study.best_trial.params["n_layers"])
        ]

        clf = MLPClassifier(
            hidden_layer_sizes=tuple(layers), random_state=runargs["seed"]
        )
    elif method == "knc":
        clf = KNeighborsClassifier(
            n_neighbors=study.best_trial.params["n_neighbors"],
            weights=study.best_trial.params["weights"],
        )
    elif method == "ada":
        clf = AdaBoostClassifier(
            n_estimators=study.best_trial.params["n_estimators"],
            learning_rate=study.best_trial.params["learning_rate"],
            random_state=runargs["seed"],
        )
    else:  # lgb
        clf = lgb.LGBMClassifier(
            application="multiclassova",
            boosting_type="dart",
            max_depth=-1,
            num_class=len(set(list(y) + list(y_ref))),
            # bagging_fraction=study.best_trial.params["bagging_fraction"],
            # feature_fraction=study.best_trial.params["feature_fraction"],
            # learning_rate=study.best_trial.params["learning_rate"],
            # max_bin=study.best_trial.params["max_bin"],
            min_data_in_leaf=study.best_trial.params["min_data_in_leaf"],
            min_data_per_group=study.best_trial.params["min_data_per_group"],
            # n_estimators=trial.suggest_int("n_estimators", 50, 200),
            # num_iterations=trial.suggest_int("num_iterations", 100, 200),
            num_leaves=study.best_trial.params["num_leaves"],
        )

    clf.fit(X, y)

    # Store features and best parameters
    features = list(X.columns)

    # Evaluate performance, if a reference was given
    if not runargs["use_reference"]:
        report = ""
    else:
        y_pred = clf.predict(X_ref)
        report = classification_report(y_ref, y_pred, zero_division=0)
        report += f"\nACCURACY SCORE: {accuracy_score(y_ref, y_pred):.4f}\n"
        report += f"\nF1 SCORE: {f1_score(y_ref, y_pred, average='weighted'):.4f}\n"

    return clf, features, study.best_trial, report


def train(
    mt_train: pd.DataFrame,
    mt_ref: Optional[pd.DataFrame],
    doculects: list,
    dataset: str,
    datafile: str,
    method: str,
    runargs: dict,
):

    # Build output filename and skip if already there and no overwrite
    serialized_file = f"classifiers/{dataset}.{datafile}.{method}.joblib"
    if not runargs["overwrite"] and Path(serialized_file).is_file():
        logging.warning(
            f"File `{serialized_file}` already exist, skipping over (use `--overwrite` if necessary)."
        )
        return

    # Train on each doculect
    classifiers = {}
    features = {}
    best_study = {}
    for doculect in doculects:
        print(f"Training `{doculect}`/`{datafile}` from `{dataset}` with {method}...")

        (
            classifiers[doculect],
            features[doculect],
            best_study[doculect],
            report,
        ) = build_classifier(mt_train, mt_ref, doculect, method, runargs)

        # Evaluate performance, if a reference was given
        if report:
            print(report)

            with open(
                f"classifiers/{dataset}.{datafile}.{doculect}.{method}.report.txt",
                "w",
                encoding="utf-8",
            ) as handler:
                handler.write(report)

    # Write classifiers and best parameters to disk
    # todo: use pathlib
    clf_dump = {
        doculect: [classifiers[doculect], features[doculect]]
        for doculect in classifiers
    }
    joblib.dump(clf_dump, serialized_file)

    with open(
        f"classifiers/{dataset}.{datafile}.{method}.best_study.txt",
        "w",
        encoding="utf-8",
    ) as handler:
        buf = {
            doculect: {
                "parameters": study.params,
                "datatime_start": str(study.datetime_start),
                "datatime_complete": str(study.datetime_complete),
                "duration": str(study.duration),
                "values": str(study.values),
            }
            for doculect, study in best_study.items()
        }

        handler.write(pprint.pformat(buf, indent=4))


def main(runargs: dict):
    """
    Script entry point.
    """
    # Get training dataframe filenames
    train_pattern = BASE_PATH / "dfs" / "*" / "training*"
    train_files = sorted(glob.glob(str(train_pattern)))

    # Take a subset, for quicker R&D, if requested
    if runargs["restricted"] == 3:
        train_files = [f for f in train_files if "listsample" in f and "0.10" in f]
    elif runargs["restricted"] == 2:
        train_files = [f for f in train_files if "listsample" in f or "allenbai" in f]
    elif runargs["restricted"] == 1:
        train_files = [
            f
            for f in train_files
            if "listsample" in f or "allenbai" in f or "davletshinaztecan" in f
        ]

    if runargs["filter"]:
        train_files = [f for f in train_files if runargs["filter"] in f]

    # Iterate over all training files
    for train_file in train_files:
        # Collect information from the filename
        path_tokens = train_file.split("/")
        datafile = path_tokens[-1]
        dataset = path_tokens[-2]

        # Read training and test data
        mt_data, doculects = common.read_mtdf_data(train_file)
        if runargs["use_reference"]:
            mt_reference, _ = common.read_mtdf_data(
                train_file.replace("training-", "test-")
            )
        else:
            mt_reference = None

        # Perform the training
        if runargs["method"] == "all":
            for method in sorted(common.METHODS):
                train(
                    mt_data, mt_reference, doculects, dataset, datafile, method, runargs
                )
        else:
            train(
                mt_data,
                mt_reference,
                doculects,
                dataset,
                datafile,
                runargs["method"],
                runargs,
            )


def parse_args() -> dict:
    """
    Parse command-line arguments and return them as a dictionary.
    """

    parser = argparse.ArgumentParser(description="Train classifiers for the ST2022.")
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="debug",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        # default="all",
        choices=common.METHODS + ["all"],
        help="Set the method for training.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Set the general pseudo-random generator seeds.",
    )
    parser.add_argument(
        "-r",
        "--restricted",
        type=int,
        default=0,
        help="Perform training on all (0), some (1), two [allenbai and listsamplesize] (2), or just listsamplesize (3).",
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        help="A substring to be used for dataset filtering (applied after restrictions, if any).",
    )
    parser.add_argument(
        "--use_reference",
        action="store_true",
        help="Use the reference data (as in the first set of ST2022) for evaluation (default is false). DON'T USE IT FOR TRAINING!",
    )
    parser.add_argument(
        "-k",
        "--kfolds",
        type=int,
        default=4,
        help="Number of k-folds for the stratified cross-validation (default: 4).",
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter optimization, not counting kfolds (default: 100).",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite files if they exist (default is to keep them).",
    )

    # Get the namespace dictionary, also for web interface compatibility
    runargs = parser.parse_args().__dict__

    return runargs


if __name__ == "__main__":
    # Make sure there are no complains when setting columns as categorical
    pd.set_option("mode.chained_assignment", None)

    # Parse command line arguments and set the logger
    args = parse_args()

    # Set the training seeds, making sure we accept strings
    new_seed = common.set_seeds(args["seed"])
    args["seed"] = new_seed

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logging.basicConfig(level=level_map[args["verbosity"]])

    main(args)
