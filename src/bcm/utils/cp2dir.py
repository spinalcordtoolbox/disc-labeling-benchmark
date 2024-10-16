"""
Based on https://github.com/neuropoly/totalspineseg/blob/main/totalspineseg/utils/cpdir.py
"""

import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
import argparse
from pathlib import Path
import shutil

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment an image using nnUNetV2 model.')
    parser.add_argument('--in-paths', help='Input files that will be copied', required=True)
    parser.add_argument('--ofolder', type=Path, help='Output folder where the files will be copied', required=True)
    parser.add_argument('--max-workers', '-w', type=int, default=mp.cpu_count(), help='Max workers to run in parallel processes, defaults to multiprocessing.cpu_count().')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Get the command-line argument values
    input_paths = args.in_paths
    ofolder = args.ofolder
    workers = args.max_workers

    cp2dir_mp(
        input_paths=input_paths,
        out_folder=ofolder,
        workers=workers
    )


def cp2dir_mp(
        input_paths,
        out_folder,
        workers=mp.cpu_count(),
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    
    input_paths = [Path(path) for path in input_paths]
    output_paths = [out_folder / path.name for path in input_paths]

    process_map(
            cp2dir,
            input_paths,
            output_paths,
            max_workers=workers,
            chunksize=1
        )


def cp2dir(
        in_path,
        out_path
    ):
    # Make sure dst directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy the file
    shutil.copy2(in_path, out_path)


if __name__=="__main__":
    main()
