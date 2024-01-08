#!/opt/conda/envs/sc-dandelion-container/bin/python
import argparse
import dandelion as ddl
import os
import scanpy
from scanpy import logging as logg

scanpy.settings.verbosity = 3


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5ddl",
        required=True,
        help=("Dandelion object to call changeo clonotypes on."),
    )
    parser.add_argument(
        "--manual_threshold",
        help=("Optional manual override of SHazaM threshold."),
    )
    parser.add_argument(
        "--plot_file",
        help=(
            "File name to save PDF of SHazaM plot to. Defaults to the input object name "
            + "with _shazam.pdf appended."
        ),
    )
    parser.add_argument(
        "--key_added",
        default="changeo_clone_id",
        help=(
            "The column name to store the identified clone ID under in the object. "
            + "Defaults to changeo_clone_id."
        ),
    )
    parser.add_argument(
        "--h5ddl_out",
        help=(
            "Path to save Dandelion object with changeo clonotypes to. Defaults to the "
            + "input object name with _changeo.h5ddl appended."
        ),
    )
    args, additional_args = parser.parse_known_args()
    # set up the default names of files if need be. needs the base name of the file
    basename = os.path.splitext(args.h5ddl)[0]
    if args.plot_file is None:
        args.plot_file = basename + "_shazam.pdf"
    if args.h5ddl_out is None:
        args.h5ddl_out = basename + "_changeo.h5ddl"
    if args.manual_threshold is not None:
        args.manual_threshold = float(args.manual_threshold)
    return args, additional_args


def main():
    """Main changeo-clonotypes."""
    logg.info("Software versions:\n")
    ddl.logging.print_header()

    start = logg.info("\nBegin assigning change-o clonotypes\n")

    # parse arguments
    args, additional_args = parse_args()

    loginfo = (
        f"--------------------------------------------------------------\n"
        f"    --h5ddl = {args.h5ddl}\n"
        f"    --manual_threshold = {str(args.manual_threshold)}\n"
        f"    --plot_file = {args.plot_file}\n"
        f"    --key_added = {args.key_added}\n"
        f"    --h5ddl_out = {args.h5ddl_out}\n"
    )
    if len(additional_args) > 0:
        additional_argsx = " ".join(additional_args)
        loginfo += f"    additional arguments:\n"
        loginfo += f"    {additional_argsx}\n"
    loginfo += (
        f": --------------------------------------------------------------\n"
    )

    logg.info(
        "command line parameters:\n",
        deep=loginfo,
    )

    # the actual process is easy. the dependencies quite a bit less so
    vdj = ddl.read_h5ddl(args.h5ddl)
    ddl.pp.calculate_threshold(
        vdj,
        manual_threshold=args.manual_threshold,
        save_plot=args.plot_file,
    )
    ddl.tl.define_clones(
        vdj, key_added=args.key_added, additional_args=additional_args
    )
    vdj.write_h5ddl(args.h5ddl_out)

    logg.info("Assigning Change-o clonotypes finished.\n", time=start)


if __name__ == "__main__":
    main()
