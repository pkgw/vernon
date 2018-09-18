# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Swiss-army-knife command-line interface to Vernon functionality.

"""
from __future__ import absolute_import, division, print_function

from pwkit import cli
from pwkit.cli import die


def farm_out_to_ljob(argv):
    "ljob is a shell script"
    import os
    from pwkit.io import Path

    if 'TOP' not in os.environ:
        die('ljob command must be run with the environment variable $TOP set to a directory')

    ljob_support_dir = (Path(__file__).parent.parent / 'ljob_support').resolve()
    os.environ['LJOB_SUPPORT'] = str(ljob_support_dir)
    ljob_script = ljob_support_dir / 'ljob.sh'
    os.execv(str(ljob_script), argv)


def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "hpc", "integrate", "ljob", "preprays", "sde", "view", "vkl"')

    if argv[1] == 'hpc':
        from ..hpc import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    elif argv[1] == 'integrate':
        from ..integrate import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    elif argv[1] == 'ljob':
        farm_out_to_ljob(['vernon ' + argv[1]] + argv[2:])
    elif argv[1] == 'preprays':
        from ..preprays import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    elif argv[1] == 'sde':
        from ..sde import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    elif argv[1] == 'view':
        from .view import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    elif argv[1] == 'vkl':
        from ..vkl import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])


def main():
    import sys
    cli.unicode_stdio()
    cli.propagate_sigint()
    cli.backtrace_on_usr1()
    entrypoint(sys.argv)
