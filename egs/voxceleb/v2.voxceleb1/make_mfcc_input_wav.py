#!/usr/bin/env python3

import os, sys
from os.path import basename, dirname, isfile, realpath, join as p_join
from glob import glob
from multiprocessing import cpu_count, Pool
from functools import partial
from shutil import copyfile

def parsing_cmd(cmd_line, prefix=None, log=None):
    cmds = cmd_line.split()
    with open(log, 'a') as log_f:
        NF = len(cmds)
        if NF == 0:
            print('Null string input for parsing_cmd()', file=log_f)
            raise Exception

        dst = p_join(prefix, cmds[0] + '.wav')
        try:
            os.remove(log_file)
        except OSError:
            pass
        if NF == 2:
            #print('cp ', cmds[1], dst)
            print('cp ', cmds[1], dst, file=log_f)
            copyfile(cmds[1], dst)
        elif NF > 3:
            if cmds[-1] == "|":
                end = len(cmds) - 2
            elif cmds[-1] == "-|":
                end = len(cmds) - 1

            #print(' '.join(cmds[1:end]) + ' ' + dst)
            print(' '.join(cmds[1:end]) + ' ' + dst, file=log_f)
            os.system(' '.join(cmds[1:end]) + ' ' + dst)

if __name__ == '__main__':
    input_prefix = sys.argv[1]
    wav_dir = sys.argv[2]

    os.makedirs('./log', exist_ok=True)
    scps = sorted(glob(input_prefix +'.*.scp'))
    for scp in scps:
        wav_prefix = realpath(p_join(wav_dir, basename(scp)[:-4]))
        os.makedirs(wav_prefix, exist_ok=True)
        log_file = p_join('./log', 'stdout_' + basename(scp))
        try:
            os.remove(log_file)
        except OSError:
            pass
        this_parsing = partial(parsing_cmd, prefix=wav_prefix, log=log_file)
        with open(scp, 'r') as scp_fd:
            scp_contents = scp_fd.read()
        scp_lines = [scp_line for scp_line in scp_contents.split('\n') if len(scp_line) > 0]
        num_of_cpus = cpu_count()  
        with Pool(num_of_cpus - 1, maxtasksperchild=100) as p:
            p.map(this_parsing, scp_lines, chunksize=5)
