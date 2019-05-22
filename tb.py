#!/usr/bin/env python

"""
Little handy script to launch tensorboard using a wildcard mask.
Courtesy of Shao-Hua Sun.
"""

import sys
import glob
import time
import psutil
import os.path
import argparse
import subprocess

from os.path import join


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def main():
    parser = argparse.ArgumentParser(description=r'Launch tensorboard on multiple directories in an easy way.')
    parser.add_argument('--port', default=6006, type=int, help='The port to use for tensorboard')
    parser.add_argument('--quiet', '-q', action='store_true', help='Run in silent mode')
    parser.add_argument('--refresh-every', '-r', dest='refresh', type=int, default=1800,
                        help='Refresh every x seconds (default 1800 sec, i.e. 30 min)')
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

    num_restarts = 0
    while True:
        num_restarts += 1
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        time.sleep(args.refresh)
        print('Kill current tensorboard process', p.pid, '...')
        kill(p.pid)
        print('Restarting tensorboard', num_restarts, '...')

    return 0


if __name__ == '__main__':
    sys.exit(main())
