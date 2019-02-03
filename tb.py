#!/usr/bin/env python

"""
Little handy script to launch tensorboard using a wildcard mask.
Courtesy of Shao-Hua Sun.
"""

import glob
import subprocess
import argparse
import os.path
import sys
from os.path import join


def main():
    parser = argparse.ArgumentParser(description=r'Launch tensorboard on multiple directories in an easy way.')
    parser.add_argument('--port', default=6006, type=int, help='The port to use for tensorboard')
    parser.add_argument('--quiet', '-q', action='store_true', help='Run in silent mode')
    parser.add_argument('filters', nargs='+', type=str, help='directories in train_dir to monitor')
    args = parser.parse_args()

    train_dirs = []
    for f in args.filters:
        matches = glob.glob(join('train_dir', f))
        for match in matches:
            if os.path.isdir(match):
                train_dirs.append(match)
                print('Monitoring', match, '...')

    train_dirs = ','.join([f'{os.path.basename(s)}:{s}' for s in train_dirs])
    cmd = f'tensorboard --port={args.port} --logdir={train_dirs}'
    if args.quiet:
        cmd += ' 2>/dev/null'

    print(cmd)
    subprocess.call(cmd, shell=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
