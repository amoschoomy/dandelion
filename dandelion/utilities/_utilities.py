#!/usr/bin/env python
import os
import re
import warnings

import numpy as np
import pandas as pd

from airr import RearrangementSchema
from collections import defaultdict
from pathlib import Path
from subprocess import run
from typing import Tuple, Union, Optional, TypeVar, List, Dict

NetworkxGraph = TypeVar("networkx.classes.graph.Graph")

TRUES = ["T", "True", "true", "TRUE", True]
FALSES = ["F", "False", "false", "FALSE", False]
HEAVYLONG = ["IGH", "TRB", "TRD"]
LIGHTSHORT = ["IGK", "IGL", "TRA", "TRG"]

# for compatibility with python>=3.10
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

# for compatibility with python==3.7
try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            """LiteralMeta class."""

            def __getitem__(self, values):
                """Return Literal."""
                if not isinstance(values, tuple):
                    values = (values,)
                return type("Literal_", (Literal,), dict(__args__=values))

        class Literal(metaclass=LiteralMeta):
            """Literal type."""

            pass


class Tree(defaultdict):
    """Create a recursive defaultdict."""

    def __init__(self, value=None):
        super(Tree, self).__init__(Tree)
        self.value = value


def dict_from_table(meta: pd.DataFrame, columns: Tuple[str, str]) -> dict:
    """
    Generate a dictionary from a dataframe.

    Parameters
    ----------
    meta : pd.DataFrame
        pandas data frame or file path
    columns : Tuple[str, str]
        column names in data frame

    Returns
    -------
    dict
        dictionary
    """
    if (isinstance(meta, pd.DataFrame)) & (columns is not None):
        meta_ = meta
        if len(columns) == 2:
            sample_dict = dict(zip(meta_[columns[0]], meta_[columns[1]]))
    elif (os.path.isfile(str(meta))) & (columns is not None):
        meta_ = pd.read_csv(meta, sep="\t", dtype="object")
        if len(columns) == 2:
            sample_dict = dict(zip(meta_[columns[0]], meta_[columns[1]]))

    sample_dict = clean_nan_dict(sample_dict)
    return sample_dict


def clean_nan_dict(d: dict) -> dict:
    """
    Remove nan from dictionary.

    Parameters
    ----------
    d : dict
        dictionary

    Returns
    -------
    dict
        dictionary with no NAs.
    """
    return {k: v for k, v in d.items() if v is not np.nan}


def flatten(l: list) -> list:
    """
    Flatten a list-in-list-in-list.

    Parameters
    ----------
    l : list
        a list-in-list list

    Yields
    ------
    list
        a flattened list.
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def makeblastdb(ref: Union[str, Path]):
    """
    Run makeblastdb on constant region fasta file.

    Wrapper for makeblastdb.

    Parameters
    ----------
    ref : Union[str, Path]
        constant region fasta file.
    """
    cmd = ["makeblastdb", "-dbtype", "nucl", "-parse_seqids", "-in", str(ref)]
    run(cmd)


def bh(pvalues: np.array) -> np.array:
    """
    Compute the Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : np.array
        array of p-values to correct

    Returns
    -------
    np.array
        np.array of corrected p-values
    """
    n = int(pvalues.shape[0])
    new_pvalues = np.empty(n)
    values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
    values.sort()
    values.reverse()
    new_values = []
    for i, vals in enumerate(values):
        rank = n - i
        pvalue, index = vals
        new_values.append((n / rank) * pvalue)
    for i in range(0, int(n) - 1):
        if new_values[i] < new_values[i + 1]:
            new_values[i + 1] = new_values[i]
    for i, vals in enumerate(values):
        pvalue, index = vals
        new_pvalues[index] = new_values[i]
    return new_pvalues


def is_categorical(array_like) -> bool:
    """Check if categorical."""
    return array_like.dtype.name == "category"


def type_check(dataframe, key) -> bool:
    """Check dtype."""
    return (
        dataframe[key].dtype == str
        or dataframe[key].dtype == object
        or is_categorical(dataframe[key])
        or dataframe[key].dtype == bool
    )


def isGZIP(filename: str) -> bool:
    """Check if is gzipped file."""
    if filename.split(".")[-1] == "gz":
        return True
    return False


def isBZIP(filename: str) -> bool:
    """Check if is bzipped file."""
    if filename.split(".")[-1] == "pbz2":
        return True
    return False


def check_filepath(
    file_or_folder_path: Union[str, Path],
    filename_prefix: Optional[str] = None,
    ends_with: Optional[str] = None,
    sub_dir: Optional[str] = None,
    within_dandelion: bool = True,
) -> Path:
    """
    Checks whether file path exists.

    Parameters
    ----------
    file_or_folder_path : Union[str, Path]
        either a string or Path object pointing to a file or folder.
    filename_prefix : Optional[str], optional
        the prefix of the filename.
    ends_with : Optional[str], optional
        the suffix of the filename. Can be flexible i.e. not just the extension.
    sub_dir : Optional[str], optional
        the subdirectory to look for the file if specified
    within_dandelion : bool, optional
        whether to look for the file within a 'dandelion' sub folder.

    Returns
    -------
    Union[Path, None]
        Path object if file is found, else None.
    """
    filename_pre = "filtered" if filename_prefix is None else filename_prefix

    ends_with = "" if ends_with is None else ends_with
    input_path = (
        Path(file_or_folder_path).expanduser()
        if str(file_or_folder_path)[0] == "~"
        else Path(file_or_folder_path)
    )
    if input_path.is_file() and str(input_path).endswith(ends_with):
        return input_path
    elif input_path.is_dir():
        if within_dandelion:
            for child in input_path.iterdir():
                if child.name[0] != ".":
                    if child.is_dir() and child.name == "dandelion":
                        out_dir = child
                        if sub_dir is not None:
                            out_dir = out_dir / sub_dir
                        for file in out_dir.iterdir():
                            if file.name[0] != ".":
                                if file.is_file() and str(file).endswith(
                                    ends_with
                                ):
                                    if file.name.startswith(
                                        filename_pre + "_contig"
                                    ):
                                        return file
        else:
            if sub_dir is not None:
                input_path = input_path / sub_dir
            for file in input_path.iterdir():
                if file.name[0] != ".":
                    if file.is_file() and str(file).endswith(ends_with):
                        if file.name.startswith(filename_pre + "_contig"):
                            return file
    else:
        return None


def cmp(a, b):
    """Python2.x cmp function."""
    return (a > b) - (a < b)


def cmp_str_emptylast(s1, s2):
    """Help sort empty string to last."""
    if not s1 or not s2:
        return bool(s2) - bool(s1)

    return cmp(s1, s2)


def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function."""

    class K:
        """Key class"""

        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            """Less than."""
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            """Greater than."""
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            """Equal."""
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            """Less than or equal."""
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            """Greater than or equal."""
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            """Not equal."""
            return mycmp(self.obj, other.obj) != 0

    return K


def not_same_call(a, b, pattern):
    """Utility function to check if a == b in terms of pattern."""
    return (re.search(pattern, a) and not re.search(pattern, b)) or (
        re.search(pattern, b) and not re.search(pattern, a)
    )


def same_call(a, b, c, pattern):
    """Utility function to check if a == b == c in terms of pattern."""
    queries = [a, b, c]
    queries = [q for q in queries if pd.notnull(q)]
    return all([re.search(pattern, x) for x in queries])


def present(x):
    """Utility function to check if x is not null or blank."""
    return pd.notnull(x) and x != ""


def check_missing(x):
    """Utility function to check if x is null or blank."""
    return pd.isnull(x) or x == ""


def all_missing(x):
    """Utility function to check if all x is not null or blank."""
    return all(pd.isnull(x)) or all(x == "")


def all_missing2(x):
    """Utility function to check if all x is not null or blank or the word None."""
    return all(pd.isnull(x)) or all(x == "") or all(x == "None")


def return_mix_dtype(data):
    """Utility function to return mixed dtypes columns."""
    check = [
        c
        for c in data.columns
        if pd.api.types.infer_dtype(data[c]).startswith("mixed")
    ]
    return check


def sanitize_data(data, ignore="clone_id"):
    """Quick sanitize dtypes."""
    data = data.astype("object")
    data = data.infer_objects()
    for d in data:
        if d in RearrangementSchema.properties:
            if RearrangementSchema.properties[d]["type"] in [
                "string",
                "boolean",
                "integer",
            ]:
                data[d].replace(
                    [None, np.nan, pd.NA, "nan", ""], "", inplace=True
                )
                if RearrangementSchema.properties[d]["type"] == "integer":
                    data[d] = [
                        int(x) if present(x) else ""
                        for x in pd.to_numeric(data[d])
                    ]
            else:
                data[d].replace(
                    [None, pd.NA, np.nan, "nan", ""], np.nan, inplace=True
                )
        else:
            if d != ignore:
                try:
                    data[d] = pd.to_numeric(data[d])
                except:
                    data[d].replace(
                        to_replace=[None, np.nan, pd.NA, "nan", ""],
                        value="",
                        inplace=True,
                    )
        if re.search("mu_freq", d):
            data[d] = [
                float(x) if present(x) else np.nan
                for x in pd.to_numeric(data[d])
            ]
        if re.search("mu_count", d):
            data[d] = [
                int(x) if present(x) else "" for x in pd.to_numeric(data[d])
            ]
    try:
        data = check_travdv(data)
    except:
        pass

    if (
        pd.Series(["cell_id", "duplicate_count", "productive"])
        .isin(data.columns)
        .all()
    ):  # sort so that the productive contig with the largest umi is first
        data.sort_values(
            by=["cell_id", "productive", "duplicate_count"],
            inplace=True,
            ascending=[True, False, False],
        )

    # check if airr-standards is happy
    validate_airr(data)
    return data


def sanitize_blastn(data):
    """Quick sanitize dtypes."""
    data = data.astype("object")
    data = data.infer_objects()
    for d in data:
        if d in RearrangementSchema.properties:
            if RearrangementSchema.properties[d]["type"] in [
                "string",
                "boolean",
                "integer",
            ]:
                data[d].replace(
                    [None, np.nan, pd.NA, "nan", ""], "", inplace=True
                )
                if RearrangementSchema.properties[d]["type"] == "integer":
                    data[d] = [
                        int(x) if present(x) else ""
                        for x in pd.to_numeric(data[d])
                    ]
            else:
                data[d].replace(
                    [None, pd.NA, np.nan, "nan", ""], np.nan, inplace=True
                )
        else:
            try:
                data[d] = pd.to_numeric(data[d])
            except:
                data[d].replace(
                    to_replace=[None, np.nan, pd.NA, "nan", ""],
                    value="",
                    inplace=True,
                )
    return data


def sanitize_data_for_saving(data):
    """Quick sanitize dtypes for saving."""
    tmp = data.copy()
    for d in tmp:
        if d in RearrangementSchema.properties:
            if RearrangementSchema.properties[d]["type"] in [
                "string",
                "boolean",
            ]:
                tmp[d].replace(
                    [None, np.nan, pd.NA, "nan", ""], "", inplace=True
                )
            if RearrangementSchema.properties[d]["type"] in [
                "integer",
                "number",
            ]:
                tmp[d].replace(
                    [None, np.nan, pd.NA, "nan", ""], np.nan, inplace=True
                )
        else:
            try:
                tmp[d] = pd.to_numeric(tmp[d])
            except:
                tmp[d].replace(
                    [None, pd.NA, np.nan, "nan", ""], "", inplace=True
                )
    return tmp


def validate_airr(data):
    """Validate dtypes in airr table."""
    tmp = data.copy()
    int_columns = []
    for d in tmp:
        try:
            tmp[d].replace(np.nan, pd.NA).astype("Int64")
            int_columns.append(d)
        except:
            pass
    bool_columns = [
        "rev_comp",
        "productive",
        "vj_in_frame",
        "stop_codon",
        "complete_vdj",
    ]
    str_columns = list(tmp.dtypes[tmp.dtypes == "object"].index)
    columns = [
        c
        for c in list(set(int_columns + str_columns + bool_columns))
        if c in tmp
    ]
    if len(columns) > 0:
        for c in columns:
            tmp[c].fillna("", inplace=True)
    for _, row in tmp.iterrows():
        contig = Contig(row).contig
        for required in [
            "sequence",
            "rev_comp",
            "sequence_alignment",
            "germline_alignment",
            "v_cigar",
            "d_cigar",
            "j_cigar",
        ]:
            if required not in contig:
                contig.update({required: ""})
    RearrangementSchema.validate_header(contig.keys())
    RearrangementSchema.validate_row(contig)


def check_travdv(data):
    """Check if locus is TRA/D."""
    data = load_data(data)
    contig = [x for x in data["sequence_id"]]
    v = [x for x in data["v_call"]]
    d = [x for x in data["d_call"]]
    j = [x for x in data["j_call"]]
    c = [x for x in data["c_call"]]
    l = [x for x in data["locus"]]
    v_dict = dict(zip(contig, v))
    d_dict = dict(zip(contig, d))
    j_dict = dict(zip(contig, j))
    c_dict = dict(zip(contig, c))
    l_dict = dict(zip(contig, l))
    for co in contig:
        if re.search("TRAV.*/DV", v_dict[co]):
            if same_call(j_dict[co], c_dict[co], d_dict[co], "TRA"):
                if not re.search("TRA", l_dict[co]):
                    l_dict[co] = "TRA"
            elif same_call(j_dict[co], c_dict[co], d_dict[co], "TRD"):
                if not re.search("TRD", l_dict[co]):
                    l_dict[co] = "TRD"
    data["locus"] = pd.Series(l_dict)
    return data


def load_data(obj: Optional[Union[pd.DataFrame, str]]) -> pd.DataFrame:
    """
    Read in or copy dataframe object and set sequence_id as index without dropping.

    Parameters
    ----------
    obj : Optional[Union[pd.DataFrame, str]]
        file path to .tsv file or pandas DataFrame object.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        if input is not found.
    KeyError
        if `sequence_id` not found in input.
    """
    if obj is not None:
        if os.path.isfile(str(obj)):
            try:
                obj_ = pd.read_csv(obj, sep="\t")
            except FileNotFoundError as e:
                print(e)
        elif isinstance(obj, pd.DataFrame):
            obj_ = obj.copy()
        else:
            raise FileNotFoundError(
                "Either input is not of <class 'pandas.core.frame.DataFrame'> or file does not exist."
            )

        if "sequence_id" in obj_.columns:
            obj_.set_index("sequence_id", drop=False, inplace=True)
            if "cell_id" not in obj_.columns:
                obj_["cell_id"] = [
                    c.split("_contig")[0] for c in obj_["sequence_id"]
                ]
        else:
            raise KeyError("'sequence_id' not found in columns of input")

        if "umi_count" in obj_.columns:
            if "duplicate_count" not in obj_.columns:
                obj_.rename(
                    columns={"umi_count": "duplicate_count"}, inplace=True
                )

        return obj_


class ContigDict(dict):
    """Class Object to extract the contigs as a dictionary."""

    def __setitem__(self, key, value):
        """Standard __setitem__."""
        super().__setitem__(key, value)

    def __hash__(self):
        """Make it hashable."""
        return hash(tuple(self))


class Contig:
    """Class Object to hold contig."""

    def __init__(self, contig, mapper=None):
        if mapper is not None:
            mapper.update({k: k for k in contig.keys() if k not in mapper})
            self._contig = ContigDict(
                {mapper[key]: vals for (key, vals) in contig.items()}
            )
        else:
            self._contig = ContigDict(contig)

    @property
    def contig(self):
        """Contig slot."""
        return self._contig


def mask_dj(data, filename_prefix, d_evalue_threshold, j_evalue_threshold):
    """Mask d/j assignment."""
    for i in range(0, len(data)):
        filePath = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_igblast_db-pass.tsv",
        )
        if filePath is not None:
            dat = load_data(filePath)
            if "d_support_blastn" in dat:
                dat["d_call"] = [
                    "" if s > d_evalue_threshold else c
                    for c, s in zip(dat["d_call"], dat["d_support_blastn"])
                ]
            if "j_support_blastn" in dat:
                dat["j_call"] = [
                    "" if s > j_evalue_threshold else c
                    for c, s in zip(dat["j_call"], dat["j_support_blastn"])
                ]

            write_airr(dat, filePath)


def write_airr(data, save):
    """Save as airr formatted file."""
    data = sanitize_data(data)
    data.to_csv(save, sep="\t", index=False)


def write_blastn(data, save):
    """Write blast output."""
    data = sanitize_blastn(data)
    data.to_csv(save, sep="\t", index=False)


## from skbio==0.5.6
def _validate_counts_vector(counts, suppress_cast=False):
    """Validate and convert input to an acceptable counts vector type.
    Note: may not always return a copy of `counts`!
    """
    counts = np.asarray(counts)
    if not np.all(np.isreal(counts)):
        raise ValueError("Counts vector must contain real-valued entries.")
    if counts.ndim != 1:
        raise ValueError("Only 1-D vectors are supported.")
    elif (counts < 0).any():
        raise ValueError("Counts vector cannot contain negative values.")

    return counts


def deprecated(details, deprecated_in, removed_in):
    """Decorator to mark a function as deprecated"""

    def deprecated_decorator(func):
        """Deprecate dectorator"""

        def deprecated_func(*args, **kwargs):
            """Deprecate function"""
            warnings.warn(
                "{} is a deprecated in {} and will be removed in {}."
                " {}".format(func.__name__, deprecated_in, removed_in, details),
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator


def format_call(
    metadata: pd.DataFrame,
    call: str,
    suffix_vdj: str = "_VDJ",
    suffix_vj: str = "_VJ",
) -> list:
    """Extract v/d/j/c call values from data."""
    call_dict = {
        "Multi": "Multi",
        "None": "None",
        "": "None",
        "unassigned": "None",
    }
    if suffix_vj is not None:
        call_1 = {
            x[0]: x[1] if present(x[1]) else "None"
            for x, y in zip(
                metadata[call + suffix_vdj].items(),
                list(metadata[call + suffix_vj]),
            )
        }
        call_2 = {
            x[0]: x[1] if present(x[1]) else "None"
            for x, y in zip(
                metadata[call + suffix_vj].items(),
                list(metadata[call + suffix_vdj]),
            )
        }
        call_2 = {x: y if "|" not in y else "Multi" for x, y in call_2.items()}
        call_4 = {
            x: "Single" if y not in call_dict else call_dict[y]
            for x, y in call_2.items()
        }
    else:
        call_1 = {
            x: y if present(y) else "None"
            for x, y in metadata[call + suffix_vdj].items()
        }
        call_2 = {x: "None" for x in call_1.keys()}
        call_4 = call_3 = call_2
    call_1 = {x: y if "|" not in y else "Multi" for x, y in call_1.items()}
    call_3 = {
        x: "Single" if y not in call_dict else call_dict[y]
        for x, y in call_1.items()
    }
    return (
        list(call_1.values()),
        list(call_2.values()),
        list(call_3.values()),
        list(call_4.values()),
    )


def format_locus(
    metadata: pd.DataFrame, suffix_vdj: str = "_VDJ", suffix_vj: str = "_VJ"
) -> pd.Series:
    """Extract locus call value from data."""
    locus_1 = dict(metadata["locus" + suffix_vdj])
    locus_2 = dict(metadata["locus" + suffix_vj])
    constant_1 = dict(metadata["isotype_status"])

    locus_dict = {}
    for i in metadata.index:
        loc1 = {
            e: l for e, l in enumerate([ll for ll in locus_1[i].split("|")])
        }
        loc2 = {
            e: l for e, l in enumerate([ll for ll in locus_2[i].split("|")])
        }
        loc1x, loc2x = [], []
        if not all([px == "None" for px in loc1.values()]):
            loc1xx = list(loc1.values())
            loc1x = [ij[:2] for ij in loc1.values()]

        if not all([px == "None" for px in loc2.values()]):
            loc2xx = list(loc2.values())
            loc2x = [ij[:2] for ij in loc2.values()]

        if len(loc1x) > 0:
            if len(list(set(loc1x))) > 1:
                tmp1 = "ambiguous"
            else:
                if len(loc1x) > 1:
                    if constant_1[i] == "IgM/IgD":
                        tmp1 = "IgM/IgD"
                    elif (all(x in ["TRB", "TRD"] for x in loc1xx)) and (
                        len(list(set(loc1xx))) == 2
                    ):
                        tmp1 = "Extra VDJ-exception"
                    else:
                        tmp1 = "Extra VDJ"
                else:
                    tmp1 = loc1xx[0]

                if len(loc2x) > 0:
                    if len(list(set(loc2x))) > 1:
                        tmp2 = "ambiguous"
                    else:
                        if len(loc2x) > 1:
                            if (all(x in ["TRA", "TRG"] for x in loc2xx)) and (
                                len(list(set(loc2xx))) == 2
                            ):
                                tmp2 = "Extra VJ-exception"
                            else:
                                tmp2 = "Extra VJ"
                        else:
                            tmp2 = loc2xx[0]
                else:
                    tmp2 = "None"

                if (
                    tmp1 not in ["None", "Extra VDJ", "Extra VDJ-exception"]
                ) and (tmp2 not in ["None", "Extra VJ", "Extra VJ-exception"]):
                    if list(set(loc1x)) != list(set(loc2x)):
                        tmp1 = "ambiguous"
                        tmp2 = "ambiguous"
        else:
            tmp1 = "None"
            if len(loc2x) > 0:
                if len(list(set(loc2x))) > 1:
                    tmp2 = "ambiguous"
                else:
                    if len(loc2x) > 1:
                        if (all(x in ["TRA", "TRG"] for x in loc2xx)) and (
                            len(list(set(loc2xx))) == 2
                        ):
                            tmp2 = "Extra VJ-exception"
                        else:
                            tmp2 = "Extra VJ"
                    else:
                        tmp2 = loc2xx[0]
            else:
                tmp2 = "None"
        if any(tmp == "ambiguous" for tmp in [tmp1, tmp2]):
            locus_dict.update({i: "ambiguous"})
        else:
            locus_dict.update({i: tmp1 + " + " + tmp2})

        if any(tmp == "None" for tmp in [tmp1, tmp2]):
            if tmp1 == "None":
                locus_dict.update({i: "Orphan " + tmp2})
            elif tmp2 == "None":
                locus_dict.update({i: "Orphan " + tmp1})
        if any(re.search("No_contig", tmp) for tmp in [tmp1, tmp2]):
            locus_dict.update({i: "No_contig"})
    result = pd.Series(locus_dict)
    return result


def sum_col(vals):
    """Sum columns if not NaN."""
    if all(pd.isnull(vals)):
        return np.nan
    else:
        return sum(vals)


def lib_type(lib: str):
    """Dictionary of acceptable loci for library type."""
    librarydict = {
        "tr-ab": ["TRA", "TRB"],
        "tr-gd": ["TRG", "TRD"],
        "ig": ["IGH", "IGK", "IGL"],
    }
    return librarydict[lib]


def movecol(
    df: pd.DataFrame,
    cols_to_move: List = [],
    ref_col: str = "",
) -> pd.DataFrame:
    """A way to order columns."""
    # https://towardsdatascience.com/reordering-pandas-dataframe-columns-thumbs-down-on-standard-solutions-1ff0bc2941d5
    cols = df.columns.tolist()
    seg1 = cols[: list(cols).index(ref_col) + 1]
    seg2 = cols_to_move

    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    return df[seg1 + seg2 + seg3]


def format_chain_status(locus_status):
    """Format chain status from locus status."""
    chain_status = []
    for ls in locus_status:
        if ("Orphan" in ls) and (re.search("TRB|IGH|TRD|VDJ", ls)):
            if not re.search("exception", ls):
                chain_status.append("Orphan VDJ")
            else:
                chain_status.append("Orphan VDJ-exception")
        elif ("Orphan" in ls) and (re.search("TRA|TRG|IGK|IGL|VJ", ls)):
            if not re.search("exception", ls):
                chain_status.append("Orphan VJ")
            else:
                chain_status.append("Orphan VJ-exception")
        elif re.search("exception|IgM/IgD", ls):
            chain_status.append("Extra pair-exception")
        elif re.search("Extra", ls):
            chain_status.append("Extra pair")
        elif re.search("ambiguous|None", ls):
            chain_status.append("ambiguous")
        else:
            chain_status.append("Single pair")
    return chain_status


def update_rearrangement_status(self):
    """Check rearrangement status."""
    if "v_call_genotyped" in self.data:
        vcall = "v_call_genotyped"
    else:
        vcall = "v_call"
    contig_status = []
    for v, j, c in zip(
        self.data[vcall], self.data["j_call"], self.data["c_call"]
    ):
        if present(v):
            if present(j):
                if present(c):
                    if len(list(set([v[:3], j[:3], c[:3]]))) > 1:
                        contig_status.append("chimeric")
                    else:
                        contig_status.append("standard")
                else:
                    if len(list(set([v[:3], j[:3]]))) > 1:
                        contig_status.append("chimeric")
                    else:
                        contig_status.append("standard")
            else:
                contig_status.append("unknown")
        else:
            contig_status.append("unknown")
    self.data["rearrangement_status"] = contig_status


def set_germline_env(
    germline: Optional[str] = None,
    org: Literal["human", "mouse"] = "human",
    input_file: Optional[Union[str, Path]] = None,
) -> Tuple[Dict, Path, Path]:
    """
    Set the paths to germline database and environment variables and relevant input files.

    Parameters
    ----------
    germline : Optional[str], optional
        path to germline database. None defaults to environmental variable $GERMLINE.
    org : Literal["human", "mouse"], optional
        organism for germline sequences.
    input_file : Optional[Union[str, Path]], optional
        path to input file.
    Returns
    -------
    Tuple[Dict, Path]
        environment dictionary and path to germline database.

    Raises
    ------
    KeyError
        if $GERMLINE environmental variable is not set.
    """
    env = os.environ.copy()
    if germline is None:
        try:
            gml = Path(env["GERMLINE"])
        except KeyError:
            raise KeyError(
                (
                    "Environmental variable $GERMLINE is missing. "
                    "Please 'export GERMLINE=/path/to/database/germlines/'"
                )
            )
        gml = gml / "imgt" / org / "vdj"
    else:
        gml = env["GERMLINE"] = Path(germline)
    if input_file is not None:
        input_file = Path(input_file)
    return env, gml, input_file


def set_igblast_env(
    igblast_db: Optional[Union[str, Path]] = None,
    input_file: Optional[Union[str, Path]] = None,
) -> Tuple[Dict, Path, Path]:
    """
    Set the igblast database and environment variables and relevant input files.

    Parameters
    ----------
    igblast_db : Optional[str], optional
        path to igblast database. None defaults to environmental variable $IGDATA.
    input_file : Optional[Union[str, Path]], optional
        path to input file.

    Returns
    -------
    Tuple[Dict, Path]
        environment dictionary and path to igblast database.

    Raises
    ------
    KeyError
        if $IGDATA environmental variable is not set.
    """
    env = os.environ.copy()
    if igblast_db is None:
        try:
            igdb = Path(env["IGDATA"])
        except KeyError:
            raise KeyError(
                (
                    "Environmental variable $IGDATA is missing. "
                    "Please 'export IGDATA=/path/to/database/igblast/'"
                )
            )
    else:
        igdb = env["IGDATA"] = Path(igblast_db)
    if input_file is not None:
        input_file = Path(input_file)
    return env, igdb, input_file


def set_blast_env(
    blast_db: Optional[str] = None,
    input_file: Optional[Union[str, Path]] = None,
) -> Tuple[Dict, Path, Path]:
    """
    Set the blast database and environment variables and relevant input files.

    Parameters
    ----------
    blast_db : Optional[str], optional
        path to blast database. None defaults to environmental variable $BLASTDB.
    input_file : Optional[Union[str, Path]], optional
        path to input file.
    Returns
    -------
    Tuple[Dict, Path]
        environment dictionary and path to igblast database.

    Raises
    ------
    KeyError
        if $BLASTDB environmental variable is not set.
    """
    env = os.environ.copy()
    if blast_db is None:
        try:
            bdb = Path(env["BLASTDB"])
        except KeyError:
            raise KeyError(
                (
                    "Environmental variable $BLASTDB is missing. "
                    "Please 'export BLASTDB=/path/to/database/blast/'"
                )
            )
    else:
        bdb = env["BLASTDB"] = Path(blast_db)
    if input_file is not None:
        input_file = Path(input_file)
    return env, bdb, input_file
