#! /bin/bash
#
# Wrapper for "ljob" processing jobs. The same wrapper is used for both the
# workers and the master (which isn't necessary, but they overlap in almost
# every way). The workers are likely in the serial_requeue partition, meaning
# that they may be canceled at any moment **and restarted**, multiple times.

if [ -z "$TOP" ] ; then
    echo >&2 "error: no enviroment variable \$TOP; improperly launched?"
    exit 1
fi

# Get some basic info down ASAP. But the place that we put the info depends on
# who we are.

export LJOB_PROC_ID="${SLURM_JOB_ID}t${SLURM_RESTART_COUNT:-0}"

if [ "$LJOB_IS_MASTER" = y ] ; then
  metadir=.
else
  metadir=bulkdata/$LJOB_PROC_ID
  mkdir $metadir || exit 1
fi

echo $SLURM_JOB_ID >$metadir/jobid # we don't know this in advance for array jobs
(date +%s ; date) >>$metadir/start.wallclock
hostname -s >>$metadir/hostname
env |sort >>$metadir/environment

# Create our scratch directory where we'll actually work.

scratch=$(mktemp -d --tmpdir=/scratch ljobXXXX)
chmod 770 $scratch
echo $scratch >$metadir/scratchdir
jobinfo=$(pwd -P)
archive=$(cd .. && pwd -P) # we're running from something like job/01.process.MMDD_HHMM
bulk=$(cd bulkdata && pwd -P)
cd $scratch
ln -s $archive archive
ln -s $jobinfo jobinfo
ln -s $bulk bulk

# Go! We need to propagate SIGTERM to the Python process if we run out of time;
# see http://veithen.github.io/2014/11/16/sigterm-propagation.html for explanation.

trap 'echo "wrapper got SIGTERM; propagating!" ; kill -TERM $pid' TERM
$TOP/launch stdbuf -eL -oL python -m pylib.ljob launch >>jobinfo/$metadir/ljob.log &
pid=$!
wait $pid
wait $pid # intentional
exitcode="$?"

# Clean up.

(date +%s ; date) >>jobinfo/$metadir/finish.wallclock
echo $exitcode >>jobinfo/$metadir/exitcode
cd /
rm -rf $scratch
exit $exitcode
