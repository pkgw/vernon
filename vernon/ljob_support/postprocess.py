#! /usr/bin/env python
# Copyright 2015 Peter Williams and collaborators
# Licensed under the MIT License
#
# Postprocess after an ljob has completed. We're run from the job's work
# directory after it has exited. It may have blown up in all sorts of ways, so
# we don't have many guarantees about the state of the workdir. Note that we
# can't import ipswich since there's no tools checkout to work with. But we
# can rely on the Anaconda Python environment.
#
# We may have to be careful about sacct info for jobs that are killed and
# requeued, though this shouldn't happen with ljob masters -- see the sacct
# "-D" option.

from __future__ import absolute_import, division, print_function, unicode_literals

import io, os, os.path, subprocess

from pwkit import Holder
from pwkit.cli import warn, die
import tweepy


def get_one_line (path):
    try:
        with io.open (path, 'rt') as f:
            return f.readline ().strip ()
    except Exception as e:
        warn ('failed to read a line from "%s": %s', path, e)
        return None


def get_lines (path):
    try:
        with io.open (path, 'rt') as f:
            for line in f:
                yield line.strip ()
    except Exception as e:
        warn ('failed to read from "%s": %s', path, e)


def count_lines (path):
    try:
        n = 0
        with io.open (path, 'rt') as f:
            for line in f:
                n += 1
        return n
    except Exception as e:
        warn ('failed to count lines of "%s": %s', path, e)
        return 0


def get_sacct_info (jobid, itemname):
    try:
        with io.open (os.devnull, 'rb') as devnull:
            info = subprocess.check_output (['sacct', '-j', str (jobid) + '.batch',
                                             '-n', '-P', '-o', itemname],
                                            shell=False, stdin=devnull, close_fds=True)
            return info.splitlines ()
    except Exception as e:
        warn ('failed to get sacct item "%s" for job %s: %s', itemname, jobid, e)
        return None


def get_sacct_first (jobid, itemname):
    lines = get_sacct_info (jobid, itemname)
    return None if lines is None else lines[0]


def get_max_worker_maxrss ():
    maxrss = 0

    for wjobid in get_lines ('worker-arraymasterids'):
        if not len (wjobid):
            continue

        lines = get_sacct_info (wjobid, 'MaxRSS')
        if lines is None:
            continue

        for line in lines:
            line = line.strip ()
            if not len (line):
                continue

            if line[-1] == 'K':
                maxrss = max (maxrss, int (line[:-1]))
            elif line[-1] == 'M':
                maxrss = max (maxrss, int (round (float (line[:-1]) * 1024)))
            else:
                warn ('unexpected sacct MaxRSS output for job %s: %r', wjobid, line)

    return maxrss


def main ():
    info = Holder ()
    info.jobname = get_one_line ('jobname.txt')
    info.jobid = get_one_line ('jobid')

    if info.jobid is None:
        info.jobid_fetch_failed = 1
        info.jobid = '?'
    else:
        line = get_sacct_first (info.jobid, 'ExitCode,MaxRSS,Elapsed,State')
        if line is None:
            info.sacct_fetch_failed = 1
            # Could theoretically fill in some of these from our various log
            # files but I can't imagine a situation where sacct will actually
            # fail on us.
            info.exitinfo = '?'
            info.mastermaxrss = '?'
            info.elapsed = '?'
            info.state = '?'
            info.success = -1
        else:
            info.exitinfo, info.mastermaxrss, info.elapsed, info.state = line.split ('|')
            info.success = 1 if info.exitinfo == '0:0' else 0

    info.workermaxrss = get_max_worker_maxrss ()

    tsubmit = get_one_line ('submit.wallclock')
    tstart = get_one_line ('start.wallclock')
    if tsubmit is not None and tstart is not None:
        info.startdelay = int (tstart) - int (tsubmit)

    try:
        info.ntasks = -1
        info.tot_nsuccess = -1
        info.tot_nfail = -1
        info.nleft = -1
        info.nattempts = -1
        info.cur_nsuccess = -1
        info.cur_nfail = -1
        natt = 0
        nsucc = 0
        nfail = 0

        info.ntasks = count_lines ('../tasks')
        info.tot_nsuccess = count_lines ('../success')
        info.tot_nfail = count_lines ('../failure')
        info.nleft = info.ntasks - info.tot_nsuccess - info.tot_nfail

        with io.open ('attempts.log', 'rt') as f:
            for line in f:
                pieces = line.strip ().split ()
                if pieces[1] == 'issued':
                    natt += 1
                elif pieces[1] == 'complete':
                    if pieces[-1] == '0':
                        nsucc += 1
                    else:
                        nfail += 1

        info.nattempts = natt
        info.cur_nsuccess = nsucc
        info.cur_nfail = nfail
    except Exception as e:
        warn ('couldn\'t summarize attempts: %s', e)

    with io.open ('postmortem.log', 'wt') as f:
        d = info.__dict__
        for k in sorted (d.iterkeys ()):
            val = d[k]
            if val is not None:
                print ('%s=%s' % (k, val), file=f)

    try:
        with io.open (os.path.expanduser ('~/.robotinfo'), 'rt') as f:
            user = f.readline ().strip ()
            consumer_key = f.readline ().strip ()
            consumer_secret = f.readline ().strip ()
            access_token = f.readline ().strip ()
            access_secret = f.readline ().strip ()

            auth = tweepy.OAuthHandler (consumer_key, consumer_secret)
            auth.set_access_token (access_token, access_secret)
            api = tweepy.API (auth)

            t = ('@' + user + ' %(jobname)s %(state)s succ=%(success)d nt=%(ntasks)d '
                 'ns=%(cur_nsuccess)d nf=%(cur_nfail)d nleft=%(nleft)d' % info.__dict__)
            api.update_status (status=t)
    except Exception as e:
        warn ('couldn\'t tweet: %s', e)


if __name__ == '__main__':
    from pwkit import cli
    cli.propagate_sigint ()
    cli.unicode_stdio ()
    main ()
