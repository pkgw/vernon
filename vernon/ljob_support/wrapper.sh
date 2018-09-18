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
ln -s $TOP top
ln -s $archive archive
ln -s $jobinfo jobinfo
ln -s $bulk bulk

# If available, unpack a local version of the software stack since I/O to the
# support files is an efficiency killer. Note that the binaries hardcode the
# install prefix, so we can only unpack it in one place. This in turn means
# that multiple workers running on the same node will share the same stack,
# and it won't be tractable to be fancy about auto-detecting outdated software
# or anything. Conversely, we do have to be fancy about different workers
# stepping on each others' toes as they get set up.

if [ -e $TOP/ljob_support/stack.tar.gz ] ; then
    mkdir -p /scratch/pwilliam/vernon

    # XXXX Temporary recovery from my busted initial efforts
    if mkdir /scratch/pwilliam/vernon/fix_busted_stack 2>/dev/null ; then
        echo "XXX won the right to remove old and busted stack"
        if [ ! -f /scratch/pwilliam/vernon/fix_busted_stack/finished ] ; then
            echo "XXX and fixing has not yet occurred"
            rm -rf /scratch/pwilliam/vernon/stack
            touch /scratch/pwilliam/vernon/fix_busted_stack/finished
        fi
    else
        for ii in {0..100} ; do
            if [ -f /scratch/pwilliam/vernon/fix_busted_stack/finished ] ; then
                break
            fi
            sleep 2
        done

        if [ ! -f /scratch/pwilliam/vernon/fix_busted_stack/finished ] ; then
            echo "XXX whooooa recovery broke?"
            exit 1
        fi
    fi

    if mkdir /scratch/pwilliam/vernon/stack 2>/dev/null ; then
        # We won the race to unpack the stack on this node.
        echo "Unpacking local stack $TOP/ljob_support/stack.tar.gz on $(hostname -s) ..."
        tar xf $TOP/ljob_support/stack.tar.gz -C /scratch/pwilliam/vernon
        touch /scratch/pwilliam/vernon/stack/unpacked.marker
    else
        for ii in {0..100} ; do
            if [ -f /scratch/pwilliam/vernon/stack/unpacked.marker ] ; then
                break
            fi
            sleep 2
        done

        if [ ! -f /scratch/pwilliam/vernon/stack/unpacked.marker ] ; then
            echo "Stack never got unpacked?? ii=$ii"
            exit 1
        fi
    fi

    export VERNON_EXTERNAL_STACK=/scratch/pwilliam/vernon/stack
fi

# Go! We need to propagate SIGTERM to the Python process if we run out of time;
# see http://veithen.github.io/2014/11/16/sigterm-propagation.html for explanation.

trap 'echo "wrapper got SIGTERM; propagating!" ; kill -TERM $pid' TERM
$TOP/launch stdbuf -eL -oL python -m vernon.ljob launch >>jobinfo/$metadir/ljob.log &
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
