# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Configuration grab-bag so that we can fully automate a standard end-to-end
prep-and-image operation on Odyssey.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
HPCConfiguration
entrypoint
'''.split()

import argparse
import os
from pwkit.cli import die
from pwkit.io import Path
import subprocess
import sys

from .config import Configuration


class HPCConfiguration(Configuration):
    """A grab-bag of configuration settings relevant to automating our expensive
    computations on an HPC cluster.

    """
    __section__ = 'hpc'

    serial_partition = 'general,shared,unrestricted'
    """Which Slurm partition serials tasks should be run on. This should not be a
    partition in which jobs can be interrupted and requeued.

    """
    ljob_worker_partition = 'general,shared,serial_requeue,unrestricted'
    "Which Slurm partition ljob worker tasks should be run on."

    preprays_n_col_groups = 2
    """How many groups the columns are broken into for preprays processing. This
    affects the tradeoff between the total number of jobs to run and how long
    each job takes to run.

    """
    preprays_n_workers = 300
    preprays_time_limit = '0-3'
    preprays_memory = '8192'

    pandi_pr_assemble_time_limit = '0-4'
    pandi_pr_assemble_memory = '16384'

    integrate_n_row_groups = 2
    """How many groups the rows are broken into for "integrate" processing. This
    affects the tradeoff between the total number of jobs to run and how long
    each job takes to run.

    """
    integrate_n_workers = 100
    integrate_time_limit = '0-3'
    integrate_memory = '8192'

    pandi_integ_assemble_time_limit = '0-1'
    pandi_integ_assemble_memory = '8192'


    def launch_ljob(self, task_name, work_dir):
        "Launches an ljob and returns the master's Slurm jobid as a string."

        n_workers = getattr(self, task_name + '_n_workers', 128)
        time_limit = getattr(self, task_name + '_time_limit', '0-1')
        memory = getattr(self, task_name + '_memory', '8192')

        argv = [
            'ljob',
            'process',
            work_dir,
            '--machine-output',
            '-n', str(n_workers),
            '-t', str(time_limit),
            '-m', str(memory),
            '-i', 'vernonhpc',
            '-p', self.ljob_worker_partition
        ]

        output = subprocess.check_output(argv, shell=False, stderr=subprocess.STDOUT)
        output = output.decode('utf8').splitlines()
        info = {}

        for line in output:
            if '=' in line:
                key, val = line.strip().split('=', 1)
                info[key] = val
            else:
                print('warning: unexpected line from "ljob process": %r' % line, file=sys.stderr)

        return info['masterjobid']


    def schedule_next_stage(self, task_name, task_argv, waitjobid):
        """Schedule a serial job to run after some other job has run.

        Returns its jobid as a string.

        """
        from shlex import quote

        time_limit = getattr(self, task_name + '_time_limit', '0-1')
        memory = getattr(self, task_name + '_memory', '8192')

        script = '#!/bin/bash\nexec ' + ' '.join(quote(a) for a in task_argv) + '\n'

        argv = [
            'sbatch',
            '-d', 'afterany:%s' % waitjobid,
            '-J', task_name,
            '--mem=%s' % memory,
            '--time=%s' % time_limit,
            '-o', '%s.log' % task_name,
            '--parsable',
            '-p', self.serial_partition,
        ]

        output = subprocess.check_output(argv, shell=False, input=script.encode('utf8'))
        output = output.decode('utf8')

        try: # sanity-check
            jobid = int(output.strip())
        except Exception:
            raise Exception('unexpected sbatch output: %r' % output)

        return str(jobid)


    def assert_job_succeeded(self, jobid):
        argv = [
            'sacct',
            '-j', '%s.batch' % jobid,
            '-n',
            '-o', 'exitcode'
        ]

        output = subprocess.check_output(argv, shell=False)
        output = output.decode('utf8')

        if output.strip() != '0:0':
            raise Exception('job %s failed: sacct reports exit code of %r' % (jobid, output))



# The standard preprays-and-integrate workflow

def make_prep_and_image_parser():
    ap = argparse.ArgumentParser(prog='vernon hpc prep-and-image')
    ap.add_argument('--stage', metavar='STAGE',
                    help='INTERNAL: workflow stage number.')
    ap.add_argument('--previd', metavar='JOBID',
                    help='INTERNAL: previous job ID')
    return ap


def prep_and_image_cli(argv):
    pre_args = argv[:2]
    settings = make_prep_and_image_parser().parse_args(args=argv[2:])
    config = HPCConfiguration.from_toml('Config.toml')

    if settings.stage is None:
        prep_and_image_ui(pre_args, settings, config)
    elif settings.stage == 'pr_assemble':
        prep_and_image_pr_assemble(pre_args, settings, config)
    elif settings.stage == 'integ_assemble':
        prep_and_image_integ_assemble(pre_args, settings, config)
    else:
        die('unknown prep-and-image stage %r', settings.stage)


def prep_and_image_ui(pre_args, settings, config):
    if not Path('Config.toml').exists():
        die('expected "Config.toml" in current directory')

    os.mkdir('preprays')
    os.mkdir('integrate')

    with open('preprays/tasks', 'wb') as tasks:
        subprocess.check_call(
            ['preprays', 'seed',
             '-c', 'Config.toml',
             '-g', str(config.preprays_n_col_groups)
            ],
            stdout = tasks,
        )

    masterid = config.launch_ljob('preprays', 'preprays')
    nextid = config.schedule_next_stage(
        'pandi_pr_assemble',
        pre_args + ['--stage=pr_assemble', '--previd=%s' % masterid],
        masterid,
    )

    print('Preprays ljob master ID:', masterid)
    print('Next-stage job ID:', nextid)

    with open('pandi_launch.log', 'wt') as log:
        print('Preprays ljob master ID:', masterid, file=log)
        print('Next-stage job ID:', nextid, file=log)


def prep_and_image_pr_assemble(pre_args, settings, config):
    config.assert_job_succeeded(settings.previd)

    # Assemble the results

    subprocess.check_call(
        ['preprays', 'assemble',
         '-c', 'Config.toml',
         'preprays/*.npy', # note: this is a glob literal
         'preprays.h5',
        ],
        shell = False,
    )

    # Seed the integration ljob

    with open('integrate/tasks', 'wb') as tasks:
        subprocess.check_call(
            ['integrate', 'seed',
             '-c', 'Config.toml',
             '-g', str(config.integrate_n_row_groups),
             'preprays.h5',
            ],
            stdout = tasks,
        )

    # Launch the integration job and its followup.

    masterid = config.launch_ljob('integrate', 'integrate')
    nextid = config.schedule_next_stage(
        'pandi_integ_assemble',
        pre_args + ['--stage=integ_assemble', '--previd=%s' % masterid],
        masterid,
    )

    print('Integrate ljob master ID:', masterid)
    print('Next-stage job ID:', nextid)


def prep_and_image_integ_assemble(pre_args, settings, config):
    config.assert_job_succeeded(settings.previd)

    # Assemble the results

    subprocess.check_call(
        ['integrate', 'assemble',
         '-c', 'Config.toml',
         'integrate/*.npy', # note: this is a glob literal
         'integrate.h5',
        ],
        shell = False,
    )

    # And we're done!

    print('Workflow completed successfully.')


# CLI driver

def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "prep-and-image"')

    if argv[1] == 'prep-and-image':
        prep_and_image_cli(argv) # note, full argv!
    else:
        die('unrecognized subcommand %r', argv[1])
