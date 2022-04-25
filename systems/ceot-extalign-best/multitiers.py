"""
Trimmed multitier implementation.

This is a trimmed down version of a private branch of the multitiers
library (https://github.com/tresoldi/multitiers), condensed and
simplified into a single Python module for the SIGTYP2022
competition.

For more information on the library, especially if you intend to use
it, please contact the author.
"""

# Import Python standard libraries
import csv
from collections import defaultdict
from typing import *

# Import 3rd-party libraries
import pandas as pd
import numpy as np

# TODO: allow to perform unicode normalization
def parse_alignment(alignment):
    """
    Parses an alignment string.

    Alignment strings are composed of space separated graphemes, with
    optional parentheses for suppressed or non-mandatory material. The
    material between parentheses is kept.

    Parameters
    ----------
    alignment : str
        The alignment string to be parsed.

    Returns
    -------
    seq : list
        A list of tokens.
    """

    return [token for token in alignment.strip().split() if token not in ["(", ")"]]


def read_wordlist(filename, sep="\t", **kwargs):
    """
    Read a wordlist as Pandas dataframe, perfoming checks.
    """

    # Set the column names that are used and carried over
    col_id = kwargs.get("col_id", "ID")
    col_doculect = kwargs.get("col_doculect", "DOCULECT")
    col_cogid = kwargs.get("col_cogid", "COGID")
    col_alignment = kwargs.get("col_alignment", "ALIGNMENT")

    # Read raw data
    df = pd.read_csv(
        filename, sep=sep, usecols=[col_id, col_doculect, col_cogid, col_alignment]
    )

    # Rename column names to internal defaults
    df = df.rename(
        columns={
            col_id: "ID",
            col_doculect: "LANG",
            col_cogid: "COGID",
            col_alignment: "ALM",
        }
    )

    # Set the index
    df = df.set_index("ID")

    # Preprocess the alignment function
    # TODO: allow more options, or at least to skip this one
    df["ALM"] = df["ALM"].apply(parse_alignment)

    # TODO: implement checks: alignments length, doculect names

    return df


def shift_tier(vector, base_name, left_orders, right_orders, oob="∅"):
    new_tiers = {}
    if all(pd.isnull(vector)):
        # If the entire vector is made of nans (= missing data), the
        # shift will by definition be composed of missing data as well;
        # this guarantees that we don't have out-of-bonds symbols along
        # with nans. We can thus just make a copy of the provided
        # vector as many times as needed.
        for left_order in left_orders:
            shifted_name = f"{base_name}_L{left_order}"
            new_tiers[shifted_name] = vector[:]
        for right_order in right_orders:
            shifted_name = f"{base_name}_R{right_order}"
            new_tiers[shifted_name] = vector[:]
    else:
        for left_order in left_orders:
            shifted_vector = vector[:-left_order]
            shifted_vector = [oob] * (
                len(vector) - len(shifted_vector)
            ) + shifted_vector
            shifted_name = f"{base_name}_L{left_order}"
            new_tiers[shifted_name] = shifted_vector

        for right_order in right_orders:
            shifted_vector = vector[right_order:]
            shifted_vector += [oob] * (len(vector) - len(shifted_vector))
            shifted_name = f"{base_name}_R{right_order}"
            new_tiers[shifted_name] = shifted_vector

    return new_tiers


# TODO: decide on gap token (should we keep `-`?)
# TODO: perform with numpy?
# TODO: have an unified mapper signature?
# TODO: rename `alignment` to `vector` whereever appropriate
def tier_mapper(alignment, model, mapper, oob=np.nan, gap="-"):

    # Return nans if data is missing
    if all(pd.isnull(alignment)):
        return [np.nan] * len(alignment)

    # todo: remove code for facilitating mapping once data is released
    # c=False
    # for token in alignment:
    #   if token not in mapper and token not in [oob,gap]:
    #       print("%s,%s" %(token, token))
    #       c=True
    # if c:
    #   alignment=["a"for token in alignment]

    # todo: nan if missing?
    # todo: gap in the mapper?
    # print([[t] for t in alignment])
    sc_vector = [mapper[token].get(model, np.nan) for token in alignment]

    return sc_vector


class MTEncoder:
    def __init__(self, mapper_file: Optional[str]):
        # Store the default mapper.
        # TODO: This is a workaround for not listing maniphono as a dependency
        if not mapper_file:
            self.mapper = None
        else:
            self.mapper = {}
            with open(mapper_file, encoding="utf-8") as handler:
                for row in csv.DictReader(handler):
                    grapheme = row.pop("GRAPHEME")
                    self.mapper[grapheme] = row

        # Information filled by the fitting method
        self.doculects = None  # all doculects, as an ordered list
        self.left = []  # all left orders, a list
        self.right = []  # all right orders, a list
        self.mapping = []  # sound class models, a list

    # TODO: list valid mappers
    def fit(self, data, mapping=None, left=None, right=None, tiers=None):
        """
        Fits an encoder to produce consistent mapping.

        The fitting procedure requires the entire data to be (eventually) used.

        Parameters
        ----------
        data : dataframe
            A dataframe with the full data to be used, such as one returned by the
            `read_wordlist()` method.
        mapping : list
            A list of the mappings to be used. Valid mappers are...
        left : int
            The left-most order to include.
        right : int
            The right-most order to include.
        """

        # Collect all doculects as an ordered set
        self.doculects = sorted(set(data["LANG"]))
        self.tiers = tiers

        if mapping:
            self.mapping = mapping

        # Set the orders, if any
        if left:
            self.left = list(range(1, left + 1))
        if right:
            self.right = list(range(1, right + 1))

    def transform(self, X):
        """
        Performs the transformation to new data.
        """

        # Distribute the data into a dictionary of `cogids`. This makes the
        # later iterations easier, and we only need to go through the
        # entire data once, also performing the normalization of alignments
        # (with potential of being computationally expansive) in the same
        # loop.
        # Note that, while it is also a bit more expansive, we copy the
        # provided data, guaranteeing that the structure provided by the
        # user is not modified.
        cogid_data = defaultdict(dict)
        for id, row in X.iterrows():
            cogid_data[row["COGID"]][row["LANG"]] = {"ALM": row["ALM"], "ID": id}

        # Add entries, once collected
        vector = defaultdict(list)
        for cogid, entries in cogid_data.items():
            # Get alignment lengths and check for consistency
            alm_lens = {len(entry["ALM"]) for entry in entries.values()}
            if len(alm_lens) > 1:
                raise ValueError(f"Cogid '{cogid}' has alignments of different sizes.")
            alm_len = list(alm_lens)[0]

            # Extend the positional tiers
            vector["index"] += [idx + 1 for idx in range(alm_len)]
            vector["index_p"] += [(idx + 1) / alm_len for idx in range(alm_len)]
            vector["rindex"] += list(range(alm_len, 0, -1))
            vector["rindex_p"] += [v / alm_len for v in range(alm_len, 0, -1)]
            vector["cogid"] += [cogid] * alm_len

            # Extend doculect vectors with alignment and id information
            for doculect in self.doculects:
                if doculect in entries:
                    alm_vector = entries[doculect]["ALM"]
                    id_vector = [entries[doculect]["ID"]] * alm_len
                else:
                    alm_vector = [np.nan] * alm_len
                    id_vector = [np.nan] * alm_len

                # Extend the doculect id tier and the segments ones (along
                # with the shifted tiers, if any)
                vector[f"id_{doculect}"] += id_vector

                vector[f"segment_{doculect}"] += alm_vector
                shifted = shift_tier(
                    alm_vector, f"segment_{doculect}", self.left, self.right
                )
                for shift_name, shift_vector in shifted.items():
                    vector[shift_name] += shift_vector

                # Extend with the sound class mapping (and shifted tiers, if any)
                for model in self.mapping:
                    sc_vector = tier_mapper(alm_vector, model, self.mapper)
                    shifted = shift_tier(
                        sc_vector,
                        f"{model}_{doculect}",
                        self.left,
                        self.right,
                    )

                    # Update vectors
                    vector[f"{model}_{doculect}"] += sc_vector
                    for shift_name, shift_vector in shifted.items():
                        vector[shift_name] += shift_vector

        # Build a dataframe from the vector(s) and filter using the provided tier
        # names, if any
        df = pd.DataFrame(vector)

        # Select tiers, if listed
        # TODO: add tiers from self.tiers if they are not in there
        if self.tiers:
            df = df[list(self.tiers)]

        # Make sure all "id_" columns are integers (we use Int32, which accepts NA)
        id_cols = [col for col in df.columns if col.startswith("id_")]
        for id_col in id_cols:
            df[id_col] = df[id_col].astype("Int32")

        return df


# todo: use normal sklearn encoders
class IndicatorEncoder:
    def __init__(self):

        self.tiers = None
        self.join = None
        self.drop_first = None

    def fit(self, tiers, drop_first=True):

        self.tiers = tiers
        self.drop_first = drop_first

    def transform(self, X):

        # Use only the requested tiers
        X = X[self.tiers]

        # build dummies
        X = pd.get_dummies(X, prefix_sep="_", drop_first=self.drop_first)

        return X
