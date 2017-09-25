# -*- mode: python; coding: utf-8 -*-
# Copyright 2015-2017 Peter Williams <peter@newton.cx> and collaborators
# Licensed under the MIT License.

"""Process a long list of tasks on a cluster using SLURM."""

from __future__ import absolute_import, division, print_function

__all__ = '''
commandline
'''.split()

# selectors is only in Python >= 3.4, so we're limiting our appeal:
import collections, io, json, os.path, random, selectors, signal
import socket, struct, subprocess, sys, time, traceback

import six
from six.moves import range

from pwkit import cli, io as pwio, Holder
from pwkit.cli import die

debug_mode = bool(len(os.environ.get('VERNON_DEBUG_LJOB', '')))

ping_interval = 30 # seconds
max_attempt_duration = 7200 # seconds
worker_idle_delay = 60 # seconds
worker_max_idle = 4200 # seconds
fails_for_give_up = 4
consec_fails_for_kill_worker = 5

@Holder
class Failcodes():
    SUCCESS = 0
    UNSPECIFIED_MASTER = 10000
    UNSPECIFIED_CLIENT = 9999
    SHUTDOWN_WHILE_WORKING = 9998
    ABORTED_BY_MASTER = 9997
    ABORTED_BY_CLIENT = 9996
    FAILED_TO_LAUNCH = 9995
    CANCELED_BY_MASTER = 9994

# Logging infrastructure

def timestamp():
    return six.text_type(time.strftime('%y/%m/%d_%H:%M:%S'))


def _log(ident, fmt, args):
    if len(args):
        text = fmt % args
    else:
        text = six.text_type(fmt)

    print(timestamp(), ident, text)
    sys.stdout.flush()

def log(fmt, *args):
    _log('--', fmt, args)

def warn(fmt, *args):
    _log('WW', fmt, args)

def logfatal(fmt, *args):
    _log('EE', fmt, args)


def _except_hook(etype, exc, tb):
    skiptb = False

    if isinstance(exc, KeyboardInterrupt):
        msg = 'interrupted by user'
        skiptb = True
    elif not isinstance(exc, EnvironmentError):
        msg = '%s: %s' % (etype.__name__, exc)
    elif hasattr(exc, 'filename') and exc.filename is not None:
        msg = '%s on %s: %s' % (etype.__name__, exc.filename, exc.strerror)
    else:
        msg = '%s: %s' % (etype.__name__, exc.strerror)

    logfatal(msg)

    if not skiptb:
        logfatal('Traceback(most recent call last):')
        for fn, line, func, text in traceback.extract_tb(tb):
            logfatal('  %s(%s:%d): %s', func, fn, line, text or '??')

    print('FATAL: %s exception: %s' % (etype.__name__, exc), file=sys.stderr)


prev_except_hook = None

def register_excepthook():
    global prev_except_hook
    prev_except_hook = sys.excepthook
    sys.excepthook = _except_hook


class LogFile(object):
    handle = None

    def __init__(self, *pathbits, **kwargs):
        if kwargs.get('append', False):
            mode = 'at'
        else:
            mode = 'wt'

        self.handle = io.open(os.path.join(*pathbits), mode, buffering=1)

    def write(self, fmt, *args):
        if len(args):
            text = fmt % args
        else:
            text = six.text_type(fmt)

        print(timestamp(), text, file=self.handle)

    def cleanup(self):
        if self.handle is None:
            return

        try:
            self.handle.flush()
            self.handle.close()
        except Exception as e:
            warn('failed to close log: %s', e)

        self.handle = None


# Robust handling of task list

class TaskList(object):
    task_ids = None
    successlog = None
    failurelog = None
    pass_id = 'undef'

    def __init__(self, taskdir):
        self.taskdir = taskdir

    def __enter__(self):
        # All-important lock file

        lockpath = os.path.join(self.taskdir, 'lock')

        try:
            lockfd = os.open(lockpath, os.O_CREAT | os.O_EXCL, 0o644)
        except Exception as e:
            logfatal('cannot acquire lock file %s', lockpath)
            raise

        os.close(lockfd)

        # Find out which tasks have already been dealt with

        nsucc = 0
        nfail = 0
        ignore_ids = set()

        for bits in pwio.pathwords(os.path.join(self.taskdir, 'success'), noexistok=True):
            nsucc += 1
            ignore_ids.add(bits[1])

        for bits in pwio.pathwords(os.path.join(self.taskdir, 'failure'), noexistok=True):
            nfail += 1
            ignore_ids.add(bits[1])

        # Read in list of all tasks

        tasks = {}
        self.task_ids = set()
        ntot = 0

        for bits in pwio.pathwords(os.path.join(self.taskdir, 'tasks')):
            ntot += 1
            ident = bits[0]
            if ident in ignore_ids:
                continue

            argv = bits[1:]
            tasks[ident] = argv
            self.task_ids.add(ident)

        log('%d tasks, %d succeeded, %d failed, %d to do',
             ntot, nsucc, nfail, len(tasks))

        # Success / failure logs

        self.successlog = LogFile(self.taskdir, 'success', append=True)
        self.failurelog = LogFile(self.taskdir, 'failure', append=True)

        return tasks # note that this is a bit funky


    def __exit__(self, etype, evalue, etb):
        try:
            if self.successlog is not None:
                self.successlog.cleanup()
                self.successlog = None

            if self.failurelog is not None:
                self.failurelog.cleanup()
                self.failurelog = None

            os.unlink(os.path.join(self.taskdir, 'lock'))
        except Exception as e:
            logfatal('swallowed error while cleaning up task list: %s', e)

        return False


    def completed(self, taskid, failcode):
        if taskid not in self.task_ids:
            warn('tried to declare completion of unknown task ID %r', taskid)
            return

        if failcode == Failcodes.SUCCESS:
            self.successlog.write('%s %s', taskid, self.pass_id)
        else:
            self.failurelog.write('%s %s %s', taskid, self.pass_id, failcode)

        self.task_ids.remove(taskid)


# Socket communication protocol -- we use explicitly delimited JSON messages
# encoded in UTF-8.

def sockwrite(sock, command, data):
    timeout = 1 # hardcoding

    content = json.dumps([command, data]).encode('utf-8')
    message = struct.pack(b'!Q', len(content)) + content
    size = len(message)
    sent = 0

    while sent < size:
        try:
            s = sock.send(message[sent:])
        except socket.error as e:
            if e.errno == 4:
                continue # EINTR: try again right now
            if e.errno != 11:
                raise # not EAGAIN: a real problem

            # EAGAIN: apparently this is the thing to do:
            sel = selectors.DefaultSelector()
            sel.register(sock, selectors.EVENT_WRITE, None)
            try:
                events = sel.select(timeout=timeout)
            except Exception as e:
                raise RuntimeError('select() failed after EAGAIN: %s' % e)

            sel.close()
            continue # we're ok to try again now

        if s == 0:
            raise RuntimeError('socket connection broken on write')
        sent += s


def make_nonblocking(stream):
    import fcntl
    fd = stream.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)


class JsonMessageReader(object):
    def __init__(self, sock):
        self.sock = sock
        self._clear()

    def _clear(self):
        """We start out by reading 8 bytes giving the length of the JSON
        message; once we have that, we read the message itself. We need to use
        the robust socket-reading code even to get the 8-byte length message
        because every so often you do get a partial read."""
        self.reading_json_length = True
        self.nleft = 8
        self.chunks = []

    def read(self):
        """Returns (command, data) or (None, None) if the message is not yet fully
        read in.

        """
        while self.nleft > 0:
            try:
                d = self.sock.recv(min(self.nleft, 2048))
            except socket.error as e:
                if e.errno == 4:
                    continue # EINTR: try again right now
                if e.errno != 11:
                    # Not EAGAIN: complain. But, for robustness, swallow the
                    # exception and hope that the mainloop eventually sees the
                    # problem if it's a showstopper.
                    warn('socket read error: %s', e)
                return None, None

            if not len(d):
                return None, None
            self.chunks.append(d)
            self.nleft -= len(d)

        data = b''.join(self.chunks)

        if self.reading_json_length:
            # We got the length of the JSON payload. But we don't actually
            # have the data yet, so toggle our switch and return "no data".
            self.reading_json_length = False
            self.nleft = struct.unpack(b'!Q', data)[0]
            self.chunks = []
            return None, None

        try:
            thing = json.loads(data.decode('utf-8'))
        except Exception:
            warn('failed to JSON decode %r', data)
            raise

        self._clear() # <- resets nleft and reading_json_length

        if len(thing) != 2 or not isinstance(thing, list):
            raise RuntimeError('expected 2-item JSON list')

        return thing


# The master job

class Task(object):
    def __init__(self, ident, argv):
        self.ident = ident
        self.argv = argv
        self.cur_attempt_id = None
        self.n_failures = 0
        self.completed = False

    def __str__(self):
        return str(self.ident)

    def available(self):
        return (not self.completed and
                self.cur_attempt_id is None and
                self.n_failures < fails_for_give_up)


class Client(object):
    def __init__(self, master, sock, address):
        self.master = master
        self.sock = sock
        self.host, self.port = address
        self.jobid = None
        self.job_array_id = None
        self.msgreader = JsonMessageReader(sock)
        self.helloed = False
        self.notified_of_exit = False
        self.sent_requeue = False
        self.last_pong = time.time()
        self.n_consec_fails = 0

        # Attributes of the current attempt:
        self.cur_task = None
        self.cur_tstart = None

        sock.setblocking(0)


    def __str__(self):
        if self.jobid is not None:
            return self.jobid
        return '%s/%d' % (self.host, self.port)


    def read(self):
        """Called when the master select() reveals that there's something to read on
        our socket. May cause us to respond to a request if we got a complete one.

        """
        try:
            command, data = self.msgreader.read()
        except Exception as e:
            warn('client %s appears to have died: %s', self, e)
            self.shutdown()
            return

        if command is None:
            return

        handler = getattr(self, 'handle_' + command)
        if handler is None:
            warn('unhandled command "%s" from client %s', command, self)
        else:
            handler(data)


    def _send(self, command, data):
        if self.sock is None:
            return

        try:
            sockwrite(self.sock, command, data)
        except Exception as e:
            warn('client %s appears to have died: %s', self, e)
            self.shutdown()


    def shutdown(self):
        """Terminate the connection and ensure that we don't try to do anything more
        with it.

        """
        if self.sock is None:
            return

        if self.cur_task is not None:
            warn('shutting down client %s while working on %s', self, self.cur_task)
            self._attempt_complete(failcode=Failcodes.SHUTDOWN_WHILE_WORKING)

        try:
            self.master.sel.unregister(self.sock)
        except Exception as e:
            warn('exception while unregistering socket for %s: %s', self, e)

        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except socket.error as e:
            warn('exception while shutting down socket for %s: %s', self, e)

        try:
            self.sock.close()
        except socket.error as e:
            warn('exception while closing socket for %s: %s', self, e)

        self.sock = None


    def maybe_signal_abort(self):
        """Tell the client to abort and terminate the connection, if we haven't done
        so already.

        """
        if self.notified_of_exit or self.sock is None:
            return

        warn('aborting client %s', self)

        if self.cur_task is not None:
            warn('had to do so while working on %s', self.cur_task)
            self._attempt_complete(failcode=Failcodes.ABORTED_BY_MASTER)

        self._send('abort', [])
        self.shutdown()
        self.notified_of_exit = True


    def maybe_requeue(self):
        """Tell SLURM to requeue the client -- to be done if the it appears to
        be on a problematic host; maybe it'll be useful again when it gets
        requeued.

        Because we want this job to keep on existing, we don't want to tell
        the client to completely exit, but we do want to cancel its current
        task.

        """
        if self.sent_requeue:
            return

        self.sent_requeue = True

        if not self.job_array_id:
            warn('want to requeue client %s, but don\'t have its array id; aborting instead', self)
            self.maybe_signal_abort()
            return

        warn('requeueing client %s: id %s', self, self.job_array_id)

        if self.cur_task is not None:
            warn('requeueing client %s while working on %s; canceling', self, self.cur_task)
            self._send('cancel_attempt', [])
            # note that attempt_complete may recurse back into maybe_requeue:
            self._attempt_complete(failcode=Failcodes.CANCELED_BY_MASTER)

        code = subprocess.call(['scontrol', 'requeue', self.job_array_id],
                                stdin=open(os.devnull, 'r'),
                                close_fds=True,
                                shell=False)
        if code:
            warn('"scontrol" invocation exited with code %d; aborting to be sure', code)
            self.maybe_signal_abort()


    def handle_hello(self, data):
        """Client registered itself."""

        if 'cookie' not in data:
            warn('client %s did not provide a cookie; dropping it', self)
            self.shutdown()
            return

        if self.master.cookie_unihex != data['cookie']:
            warn('client %s provided the wrong cookie; dropping it', self)
            self.shutdown()
            return

        log('client %s is jobid %s, array_id %s', self, data['jobid'], data['job_array_id'])
        self.jobid = data['jobid']
        self.job_array_id = data['job_array_id'] or None
        self.helloed = True


    def handle_request(self, data):
        """Client wants a new task to do."""

        if not self.helloed:
            warn('client %s may not make a request before saying hello', self)
            return

        if self.cur_task is not None:
            warn('client %s requested another task before finishing '
                  'its previous one', self)
            return

        if self.sent_requeue:
            warn('client %s wants a task but I\'m trying to get it '
                  'requeued; ignoring', self)
            return

        available = [i for i in self.master.tasks if i.available()]

        if self.master.no_more():
            # All done! Let the worker know
            log('notifying client %s that it can exit', self)
            self.notified_of_exit = True
            command = 'alldone'
            data = []
        elif len(available):
            # We can offer something to work on.
            task = available[random.randrange(len(available))]
            command = 'task'
            data = self._attempt_task(task)
        else:
            # Nothing available right now, but maybe something new will come
            # up -- possible if attempts fail frequently. Which they
            # shouldn't, but ...
            command = 'idle'
            data = []

        self._send(command, data)


    def handle_complete(self, data):
        """Client finished its current task. Presumably it will want a new one now,
        but actually requesting it is a separate command.

        """
        if not self.helloed:
            warn('client %s may not report finishing before saying hello', self)
            return

        if self.cur_task is None:
            warn('client %s claimed to finish a task but it has none active', self)
            return

        self._attempt_complete(failcode=data['failcode'])


    def handle_pong(self, data):
        """Client saw our ping."""

        self.last_pong = time.time()


    def do_housekeeping(self):
        """Ping the client if we haven't done so recently and check whether
        its attempt is taking too long.

        """
        if self.notified_of_exit or self.sock is None:
            return

        if time.time() > self.last_pong + ping_interval:
            self._send('ping', [])

        if self.cur_task is not None:
            if time.time() > self.cur_tstart + max_attempt_duration:
                # It might be better to just treat this as a single attempt failure,
                # but that would mean that a slow worker would have to take too
                # long on five consecutive tasks(or otherwise fail them) to ever
                # get requeued. It seems better to aggressively requeue under the
                # assumption that the worker host is the problem.
                warn('client %s took too long in attempt %s of task %s; requeueing',
                      self, self.cur_task.cur_attempt_id, self.cur_task)
                self.maybe_requeue()


    def _attempt_task(self, task):
        """Housekeeping for when the client is going to attempt a new task."""

        self.cur_task = task
        task.cur_attempt_id = self.master.next_attempt_id
        self.master.next_attempt_id += 1
        self.cur_tstart = time.time()

        self.master.attlog.write('issued   job/att/task %s %s %s',
                                  self, task.cur_attempt_id, task.ident)

        return {
            'argv': task.argv,
            'attemptid': task.cur_attempt_id,
            'taskid': task.ident,
        }


    def _attempt_complete(self, failcode=Failcodes.UNSPECIFIED_MASTER):
        """Housekeeping for when the client has finished its attempt."""

        task = self.cur_task
        elapsed = time.time() - self.cur_tstart

        self.master.attlog.write('complete job/att/task/elap/failcode %s %d %s %d %d',
                                  self, task.cur_attempt_id, task, elapsed,
                                  failcode)

        if failcode == Failcodes.SUCCESS:
            task.completed = True
            self.n_consec_fails = 0
            self.master.ncompleted += 1
            self.master.tasklist.completed(task.ident, failcode)
        else:
            warn('attempt %s of task %s on worker %s failed',
                  task.cur_attempt_id, task, self)

            task.n_failures += 1
            if task.n_failures >= fails_for_give_up:
                warn('hit %d failures; giving up on it', task.n_failures)
                self.master.ngivenup += 1
                self.master.tasklist.completed(task.ident, failcode)

            self.n_consec_fails += 1

        task.cur_attempt_id = None
        self.cur_task = None
        self.cur_tstart = None

        if self.n_consec_fails >= consec_fails_for_kill_worker:
            # Do this check after clearing cur_task; otherwise maybe_requeue
            # re-calls this function.
            warn('worker %s has had %d consecutive task failures', self, self.n_consec_fails)
            self.maybe_requeue()



class MasterState(object):
    def __init__(self, jobid, tasklist, todo):
        self.jobid = jobid
        self.tasklist = tasklist
        self.sock = None
        self.attlog = None
        self.clients = {}
        self.tasks = []
        self.next_attempt_id = 0
        self.ncompleted = 0
        self.ngivenup = 0
        self.max_attempts = None
        self.prev_sigterm = None
        self.terminate_signaled = False

        for ident, argv in six.iteritems(todo):
            self.tasks.append(Task(ident, argv))

        if six.PY2:
            self.cookie_unihex = unicode(os.urandom(32).encode('hex'))
        else:
            self.cookie_unihex = os.urandom(32).hex()



    def __enter__(self):
        self.prev_sigterm = signal.signal(signal.SIGTERM, self.on_sigterm)
        self.sel = selectors.DefaultSelector()
        return self


    def __exit__(self, etype, evalue, etb):
        log('cleaning up and exiting')

        if self.sel is not None:
            self.sel.close()
            self.sel = None

        if self.prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self.prev_sigterm)
            self.prev_sigterm = None

        if self.sock is not None:
            try:
                self.sock.close()
            except Exception as e:
                warn('failed to shut down socket: %s', e)
            self.sock = None

        if self.attlog is not None:
            self.attlog.cleanup()

        return False


    def setup(self):
        with io.open('jobinfo/passid.txt', 'rt') as f:
            self.tasklist.pass_id = f.readline().strip()

        self.attlog = LogFile('jobinfo/attempts.log')

        # Artificial limit on number of attempts?
        for line in pwio.pathlines('jobinfo/maxattempts.txt', noexistok=True):
            self.max_attempts = int(line)
            break

        # Regex filter on attempts to consider?
        for line in pwio.pathlines('jobinfo/taskidregex.txt', noexistok=True):
            import re
            line = line.rstrip()
            filter = re.compile(line)
            self.tasks = [t for t in self.tasks
                          if len(filter.findall(t.ident))]
            log('filtered tasks by regex %r; %d left', line, len(self.tasks))
            break

        # Get a socket
        if debug_mode:
            host = ''
        else:
            host = socket.gethostname()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(0)
        sock.bind((host, 0)) # auto-chooses an available port
        sock.listen(5)
        dest, port = sock.getsockname()
        log('listening on %s/%d', dest, port)
        self.sock = sock
        self.sel.register(sock, selectors.EVENT_READ, self.accept_new_client)

        # Write socket info such that only people with our UID can read it.
        # However, anyone can connect to the socket, so we require a random
        # code for a small layer of extra safety. We're likely on a network
        # filesystem and we send the cookie on an insecure TCP connection, so
        # this isn't exactly unbreakable.

        fd = os.open('jobinfo/sockinfo.txt', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)

        with io.open(fd, 'wt') as f:
            print(six.text_type(dest), file=f)
            print(six.text_type(port), file=f)
            print(self.cookie_unihex, file=f)


    def accept_new_client(self):
        try:
            csock, address = self.sock.accept()
        except socket.error as e:
            warn('accept fail: %s', e)
            return

        log('new client connection from %r', address)
        client = Client(self, csock, address)
        self.clients[csock] = client
        self.sel.register(csock, selectors.EVENT_READ, lambda: client.read())


    def on_sigterm(self, signum, stack):
        """In the SLURM context, this probably means that we've run out of
        time and need to wrap up ASAP. (Odyssey gives us 30 seconds.) Tell the
        workers to abort and bail. Except, since this event may happen
        asynchronously, just raise a flag and actually handle things in the
        main loop."""
        warn('SIGTERM received, aborting ASAP')
        self.terminate_signaled = True


    def no_more(self):
        if self.terminate_signaled:
            return True
        if self.max_attempts is not None and self.next_attempt_id >= self.max_attempts:
            return True
        return (self.ncompleted + self.ngivenup == len(self.tasks))


    def mainloop(self):
        timeout = 10
        next_status = time.time() + 10

        while True:
            # Forced immediate termination?
            if self.terminate_signaled:
                for c in six.itervalues(self.clients):
                    if not c.notified_of_exit:
                        c.maybe_signal_abort()
                break

            # All done?
            can_quit = self.no_more()

            if can_quit:
                for c in six.itervalues(self.clients):
                    if not c.notified_of_exit:
                        can_quit = False
                        break

            if can_quit:
                break

            # Nope, still stuff to do. Make sure everybody's still alive.
            for c in six.itervalues(self.clients):
                c.do_housekeeping()

            # Dead clients? The ping may have revealed that one died.
            for s, c in list(self.clients.items()):
                # note: not six.iteritems() since we might modify the dict:
                if c.sock is None:
                    del self.clients[s] # this client died

            # Check for messages.
            for key, mask in self.sel.select(timeout=timeout):
                key.data() # this is the callback lambda we've set up

            # Possibly issue a status message.
            now = time.time()
            if now > next_status:
                log('%d processed, %d given up, %d attempts, %d clients',
                     self.ncompleted, self.ngivenup, self.next_attempt_id,
                     len(self.clients))
                next_status = now + 300

        log('shutting down; %d of %d tasks processed in %d attempts; %d aborted',
             self.ncompleted, len(self.tasks), self.next_attempt_id, self.ngivenup)

        for client in six.itervalues(self.clients):
            try:
                client.shutdown()
            except Exception as e:
                warn('an error occurred while trying to shut down %s/%d',
                      client.host, client.port)


def launch_master(jobid):
    tasklist = TaskList('archive')

    with tasklist as todo: # meaning of this is a bit funky; todo is a dict
        with MasterState(jobid, tasklist, todo) as state:
            state.setup()
            state.mainloop()


# A worker job

class WorkerState(object):
    def __init__(self, jobid):
        self.jobid = jobid
        self.sock = None
        self.attlog = None
        self.stdiolog = None
        self.more_tasks = True
        self.need_request = True
        self.next_request_time = None
        self.prev_sigterm = None
        self.terminate_signaled = False
        self.last_completion_time = None

        self.cur_attempt_id = None
        self.cur_task_id = None
        self.attproc = None

        # This is the job id that we need to use if we're going to requeue
        # this job:
        array_jid = os.environ.get('SLURM_ARRAY_JOB_ID', '')
        array_tid = os.environ.get('SLURM_ARRAY_TASK_ID', '')
        if not len(array_jid) or not len(array_tid):
            warn('missing SLURM_ARRAY_*_ID environment variables')
            self.job_array_id = ''
        else:
            self.job_array_id = array_jid + '_' + array_tid


    def __enter__(self):
        self.prev_sigterm = signal.signal(signal.SIGTERM, self.on_sigterm)
        return self


    def __exit__(self, etype, evalue, etb):
        if self.prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self.prev_sigterm)
            self.prev_sigterm = None

        if self.attproc is not None:
            warn('shouldn\'t-happen: exiting with subprocess still running?')
            self.abort_attempt(Failcodes.ABORTED_BY_CLIENT)

        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
            except Exception as e:
                warn('failed to shut down socket: %s', e)
            self.sock = None

        if self.attlog is not None:
            self.attlog.cleanup()

        if self.stdiolog is not None:
            try:
                self.stdiolog.flush()
                self.stdiolog.close()
            except Exception as e:
                warn('failed to close stdio log: %s', e)
            self.stdiolog = None

        return False


    def connect(self):
        self.attlog = LogFile('bulk/%s/attempts.log' % self.jobid)
        self.stdiolog = io.open('bulk/%s/stdio.log' % self.jobid, 'wt', buffering=1)

        dest = port = cookie = None

        for i in range(6):
            try:
                with io.open('jobinfo/sockinfo.txt', 'rt') as f:
                    dest = f.readline().strip()
                    port = int(f.readline())
                    cookie = f.readline().strip()
                break
            except Exception as e:
                warn('failed to read master socket info: %s', e)
            time.sleep(10)

        if cookie is None:
            die('took too long to find master; giving up')

        log('connecting to %s/%d', dest, port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((dest, port))
        sock.setblocking(0)
        self.sock = sock
        self.msgreader = JsonMessageReader(sock)
        self.send('hello', {
            'jobid': six.text_type(self.jobid),
            'cookie': six.text_type(cookie),
            'job_array_id': six.text_type(self.job_array_id),
        })


    def on_sigterm(self, signum, stack):
        """See MasterState.on_sigterm(). We have 30 seconds to live."""
        warn('SIGTERM received, aborting ASAP')
        self.terminate_signaled = True
        self.more_tasks = False


    def master_died(self, exc):
        logfatal('aborting: the master appears to have died: %s', exc)

        try:
            self.sock.close()
        except socket.error as e:
            warn('exception while closing socket: %s', e)

        self.more_tasks = False
        self.sock = None
        self.abort_attempt(Failcodes.ABORTED_BY_CLIENT, notify_master=False)


    def send(self, command, data):
        if self.sock is None:
            return

        try:
            sockwrite(self.sock, command, data)
        except Exception as e:
            self.master_died(e)


    def _attempt_complete(self, failcode=Failcodes.UNSPECIFIED_CLIENT, notify_master=True):
        self.last_completion_time = time.time()
        elapsed = self.last_completion_time - self.attempt_start_time

        self.stdiolog.seek(0, 2) # get to end of output(actually needed?)
        self.attlog.write('complete att/task/ofs/elap/failcode %s %s %d %d %d',
                           self.cur_attempt_id, self.cur_task_id,
                           self.stdiolog.tell(), elapsed, failcode)
        print('\n-------- %s att=%s task=%s elap=%d failcode=%d --------' %
              (timestamp(), self.cur_attempt_id, self.cur_task_id,
                elapsed, failcode), file=self.stdiolog)

        if notify_master:
            self.send('complete', {'failcode': failcode})

        self.need_request = True
        self.next_request_time = None
        self.attproc = None


    def abort_attempt(self, failcode, notify_master=True):
        if self.attproc is None:
            return

        try:
            os.kill(self.attproc.pid, 3) # SIGQUIT
            os.kill(self.attproc.pid, 9) # SIGKILL
        except Exception as e:
            # Probably a "no such process" error, which is fine -- the attempt
            # process actually *is* dead, which is what we want anyway.
            warn('couldn\'t kill attempt proc(PID %d) while aborting: %s',
                  self.attproc.pid, e)

        self._attempt_complete(failcode=failcode, notify_master=notify_master)


    def handle_task(self, data):
        if self.attproc is not None:
            warn('got task from server even though we already have one')
            return

        self.cur_attempt_id = data['attemptid']
        self.cur_task_id = data['taskid']
        self.attempt_start_time = time.time()

        try:
            self.stdiolog.seek(0, 2) # make sure we're at end of stdio log
            print('-------- %s att=%s task=%s argv="%s" --------\n' %
                  (timestamp(), data['attemptid'], data['taskid'],
                    ' '.join(data['argv'])), file=self.stdiolog)
            self.stdiolog.flush()
            self.attlog.write('starting att/task/ofs %s %s %d',
                               data['attemptid'], data['taskid'],
                               self.stdiolog.tell())

            self.attproc = subprocess.Popen(data['argv'],
                                            stdin = open(os.devnull, 'rb'),
                                            stdout = self.stdiolog,
                                            stderr = subprocess.STDOUT,
                                            shell = False,
                                            close_fds = False)

            # don't need to ask for a task until we're done with this one!
            self.need_request = False
            self.next_request_time = None
        except Exception as e:
            warn('failed to launch process for task %r: %s', data, e)
            self._attempt_complete(failcode=Failcodes.FAILED_TO_LAUNCH)


    def handle_idle(self, data):
        # Ask again after waiting for a bit
        self.next_request_time = time.time() + worker_idle_delay


    def handle_alldone(self, data):
        if self.attproc is not None:
            warn('got alldone from server even though we\'re not!')
        log('got alldone signal')
        self.more_tasks = False


    def handle_abort(self, data):
        log('got abort signal')
        self.abort_attempt(Failcodes.ABORTED_BY_CLIENT, notify_master=False)
        self.terminate_signaled = True
        self.more_tasks = False


    def handle_cancel_attempt(self, data):
        log('got cancel_attempt signal')
        self.abort_attempt(Failcodes.CANCELED_BY_MASTER, notify_master=False)


    def handle_ping(self, data):
        self.send('pong', data)


    def mainloop(self):
        from select import select, error as selecterror
        timeout = 2

        while self.more_tasks or self.attproc is not None:
            # Forced termination?
            if self.terminate_signaled:
                self.abort_attempt(Failcodes.ABORTED_BY_CLIENT)
                break

            # Finished current task?
            if self.attproc is not None:
                retcode = self.attproc.poll()

                if retcode is None:
                    pass # still going
                else:
                    self._attempt_complete(failcode=retcode)

            # Need to request a new task?
            if (self.next_request_time is not None and time.time() > self.next_request_time):
                self.need_request = True
                self.next_request_time = None

            if self.need_request:
                self.send('request', [])
                # bother the master later if we don't hear anything:
                self.need_request = False
                self.next_request_time = time.time() + worker_idle_delay

            if self.sock is None:
                break # can get here if the send('request') triggers master_died()

            if self.last_completion_time is not None and self.attproc is None:
                if time.time() > self.last_completion_time + worker_max_idle:
                    # I've seen cases where workers linger after the master
                    # dies, for unclear reasons. No attempt should take longer
                    # than an hour, so if we haven't finished anything for
                    # longer than that, something's wrong.
                    warn('I haven\'t had anything to do in a really long time; quitting')
                    break

            files = [self.sock]
            try:
                readable, writable, errored = select(files, [], files, timeout)
            except selecterror as e:
                warn('error in select(): %s', e)
                readable = writable = errored = ()

            if len(readable):
                try:
                    command, data = self.msgreader.read()
                except Exception as e:
                    self.master_died(e)
                    return

                if command is not None:
                    handler = getattr(self, 'handle_' + command)
                    if handler is None:
                        warn('unexpected command "%s" from master', command)
                    else:
                        handler(data)

            if len(errored):
                self.master_died('<unspecified select() socket error>')
                return


def launch_worker(jobid):
    if debug_mode:
        time.sleep(1) # give master time to get going

    with WorkerState(jobid) as state:
        state.connect()
        state.mainloop()


# Command-line driver

def do_launch(argv):
    """We're a master or worker job, which we can distinguish from the
    environment variable LJOB_IS_MASTER. (We could avoid needing that
    environment variable, but it's easy and fine that way.) We take a unique
    identifier from the LJOB_PROC_ID variable (which allows us not to have to
    worry about what particular mechanism was used to launch us).

    """
    ismaster = os.environ.get('LJOB_IS_MASTER')
    if ismaster is None:
        die('framework failure: no environment variable LJOB_IS_MASTER')

    jobid = os.environ.get('LJOB_PROC_ID')
    if jobid is None:
        die('framework failure: no environment variable LJOB_PROC_ID')

    if ismaster == 'y':
        launch_master(jobid)
    else:
        launch_worker(jobid)


def commandline(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        die('usage: {ljob} launch')

    command = argv[1]

    if command == 'launch':
        do_launch(argv)
    else:
        die('unrecognized subcommand "%s"', command)


# Helpers for working with logs

def _dump_worker_attempt_log(jobdir, workerid, attid):
    startofs = endofs = elapsed = failcode = None

    with io.open(os.path.join(jobdir, 'bulkdata', workerid, 'attempts.log'), 'rt') as f:
        for line in f:
            pieces = line.split()

            if pieces[3] != attid:
                continue

            if pieces[1] in('starting', 'complete'):
                if pieces[1][0] == 's':
                    startofs = int(pieces[5])
                else:
                    endofs = int(pieces[5])
                    elapsed = int(pieces[6])
                    failcode = int(pieces[7])

                if startofs is not None and endofs is not None:
                    break

    if startofs is None or endofs is None:
        die('cannot find offset information in logs of worker %s', workerid)

    ntoread = endofs - startofs

    print('==> attempt %s worker=%s elapsed=%d failcode=%d <==' %
          (attid, workerid, elapsed, failcode))

    with io.open(os.path.join(jobdir, 'bulkdata', workerid, 'stdio.log'), 'rb') as f:
        f.seek(startofs, 0)

        while ntoread > 0:
            d = f.read(min(ntoread, 4096))
            sys.stdout.write(d)
            ntoread -= len(d)


def dump_task_log(jobdir, taskid):
    """Helper function for extracting the log for a given attempt from a queue-job
    work directory. Used by the 'ljob tasklog' command, so this is written
    in a completely non-library style. Will dump output from all relevant
    attempts.

    """
    taskid = six.text_type(taskid)
    attempts = []
    workerid = None

    with io.open(os.path.join(jobdir, 'attempts.log'), 'rt') as f:
        for line in f:
            pieces = line.split()

            if pieces[1] != 'issued':
                continue
            if pieces[5] == taskid:
                attempts.append((pieces[3], pieces[4])) # worker ID, attempt ID

    if not len(attempts):
        die('cannot find any attempts for task %s', taskid)

    _dump_worker_attempt_log(jobdir, *attempts[0])

    for attinfo in attempts[1:]:
        print()
        _dump_worker_attempt_log(jobdir, *attinfo)


def _dump_failed_worker_attempts(atts, jobdir, workerid, first=False):
    startofs = endofs = elapsed = failcode = None

    with io.open(os.path.join(jobdir, 'bulkdata', workerid, 'stdio.log'), 'rb') as stdio:
        for line in atts:
            pieces = line.split()

            if pieces[1] == 'starting':
                startofs = int(pieces[5])
                continue

            if pieces[1] != 'complete':
                continue

            failcode = int(pieces[7])

            if failcode == Failcodes.SUCCESS:
                continue

            attid = pieces[3]
            taskid = pieces[4]
            endofs = int(pieces[5])
            elapsed = int(pieces[6])
            ntoread = endofs - startofs

            if first:
                first = False
            else:
                print()

            print('==> attempt %s task=%s worker=%s elapsed=%d failcode=%d <==' %
                  (attid, taskid, workerid, elapsed, failcode))

            if ntoread == 0:
                print('[no output]')
            else:
                stdio.seek(startofs, 0)

                while ntoread > 0:
                    d = stdio.read(min(ntoread, 4096))
                    sys.stdout.write(d)
                    ntoread -= len(d)

            startofs = endofs = elapsed = failcode = ntoread = None

    return first


def dump_failed_attempt_logs(jobdir):
    first = True
    anyok = False

    for workerid in os.listdir(jobdir):
        try:
            with io.open(os.path.join(jobdir, 'bulkdata', workerid, 'attempts.log'), 'rt') as atts:
                anyok = True
                first = _dump_failed_worker_attempts(atts, jobdir, workerid, first=first)
        except IOError:
            pass # assume ENOENT

    if not anyok:
        warn('found no attempts.log files in subdirectories; wrong jobdir?')


def _word_after(haystack, needle):
    try:
        idx = haystack.index(needle)
        return haystack[idx + len(needle):].split()[0]
    except Exception:
        return '???'


def _grep_worker_attempt_log(jobdir, workerid, regex):
    cur_att = 'a=???'
    cur_task = 't=???'

    try:
        with io.open(os.path.join(jobdir, 'bulkdata', workerid, 'stdio.log'), 'rt') as f:
            for line in f:
                if line.startswith('-------- '):
                    if 'argv' in line:
                        # Delimiter indicating new record
                        cur_att = 'a=' + _word_after(line, 'att=')
                        cur_task = 't=' + _word_after(line, 'task=')
                    continue

                if len(regex.findall(line)):
                    print('%12s %8s %8s:' % (workerid, cur_task, cur_att), line, end='')
    except IOError:
        # grep_attempt_log() calls us for every item in the work directory
        # regardless of whether it is actually a worker info directory
        pass


def _ungrep_worker_attempt_log(jobdir, workerid, regex):
    cur_att = 'a=???'
    cur_task = 't=???'
    matched = True # prevents output when the first delimiter is seen
    anyoutput = False

    try:
        with io.open(os.path.join(jobdir, 'bulkdata', workerid, 'stdio.log'), 'rt') as f:
            for line in f:
                if line.startswith('-------- '):
                    if 'argv' in line:
                        # Delimiter indicating new record
                        if not matched and anyoutput:
                            print('%12s %8s %8s' % (workerid, cur_task, cur_att))

                        cur_att = 'a=' + _word_after(line, 'att=')
                        cur_task = 't=' + _word_after(line, 'task=')
                        matched = False
                        anyoutput = False
                    continue

                if len(regex.findall(line)):
                    matched = True
                if len(line.strip()):
                    anyoutput = True

            if not matched and anyoutput:
                # Last line may be special since it often corresponds to a
                # worker that was killed mid-task.
                print('%12s %8s %8s *' % (workerid, cur_task, cur_att))
    except IOError:
        # grep_attempt_log() calls us for every item in the work directory
        # regardless of whether it is actually a worker info directory
        pass


def grep_attempt_log(jobdir, regex_str, mode='grep'):
    """Helper function for grepping through attempt logs. Used by the 'ljob
    att(un)grep' commands, so this is written in a completely non-library
    style.

    """
    import re

    try:
        regex = re.compile(regex_str)
    except Exception as e:
        die('cannot compile regular expression %r: %s', regex_str, e)

    if mode == 'grep':
        func = _grep_worker_attempt_log
    elif mode == 'ungrep':
        func = _ungrep_worker_attempt_log
    else:
        die('internal bug, unknown mode %r', mode)

    for workerid in os.listdir(os.path.join(jobdir, 'bulkdata')):
        func(jobdir, workerid, regex)


# Command-line driver

if __name__ == '__main__':
    cli.propagate_sigint()
    register_excepthook()
    commandline()
