from collections import defaultdict
from pathlib import Path


class Tree(defaultdict):
    """Create a recursive defaultdict."""

    def __init__(self, value=None):
        super(Tree, self).__init__(Tree)
        self.value = value


def fasta_iterator(fh: str | Path) -> tuple[str, str]:
    """
    Read in a fasta file as an iterator.

    Parameters
    ----------
    fh : str | Path
        Path to fasta file.

    Returns
    -------
    tuple[str, str]
        Fasta header and sequence.

    Yields
    ------
    Iterator[tuple[str, str]]
        Fasta header and sequence.
    """
    while True:
        line = fh.readline()
        if line.startswith(">"):
            break
    while True:
        header = line[1:-1].rstrip()
        sequence = fh.readline().rstrip()
        while True:
            line = fh.readline()
            if not line:
                break
            if line.startswith(">"):
                break
            sequence += line.rstrip()
        yield (header, sequence)
        if not line:
            return


def write_fasta(
    fasta_dict: dict[str, str], out_fasta: str | Path, overwrite=False
):
    """
    Generic fasta writer using fasta_iterator

    Parameters
    ----------
    fasta_dict : dict[str, str]
        Dictionary containing fasta headers and sequences as keys and records respectively.
    out_fasta : str | Path
        Path to write fasta file to.
    overwrite : bool, optional
        Whether or not to overwrite the output file (out_fasta), by default False.
    """
    if overwrite:
        fh = open(out_fasta, "w")
        fh.close()
    out = ""
    for l in fasta_dict:
        out = ">" + l + "\n" + fasta_dict[l] + "\n"
        write_output(out, out_fasta)


def write_output(out: str, file: str | Path):
    """
    General line writer.

    Parameters
    ----------
    out : str
        String to write to file.
    file : str | Path
        Path to file to write to.
    """
    fh = open(file, "a")
    fh.write(out)
    fh.close()
