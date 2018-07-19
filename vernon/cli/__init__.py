# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Swiss-army-knife command-line interface to Vernon functionality.

"""
from __future__ import absolute_import, division, print_function

from pwkit import cli
from pwkit.cli import die


def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "hpc", "integrate", "ljob", "neuro", "preprays", "sde", "view", "vkl"')

    if argv[1] == 'hpc':
        from .hpc import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    elif argv[1] == 'integrate':
        from .integrate import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    elif argv[1] == 'ljob':
        die('IMPLEMENT ME')
    elif argv[1] == 'neuro':
        from .neurosynchro.cli import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    elif argv[1] == 'preprays':
        from .preprays import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    elif argv[1] == 'sde':
        from .sde import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    elif argv[1] == 'view':
        from .cli.view import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    elif argv[1] == 'vkl':
        from .vkl import entrypoint
        entrypoint(['vernon ' + argv[1]] + argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])


def main():
    import sys
    cli.unicode_stdio()
    cli.propagate_sigint()
    cli.backtrace_on_usr1()
    entrypoint(sys.argv)
