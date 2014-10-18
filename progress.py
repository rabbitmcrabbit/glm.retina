"""
=================================================
Module for determining the progress of a for loop
=================================================
"""

import time
import sys
from numpy import floor, max


def time_elapsed(initial_time):
    """Prints the time elapsed since an initial time."""
    dt = time.gmtime(time.time() - initial_time)
    if dt.tm_hour > 0:
        str = '%dh%dm%ds' % (dt.tm_hour, dt.tm_min, dt.tm_sec)
    elif dt.tm_min > 0:
        str = '%dm%ds' % (dt.tm_min, dt.tm_sec)
    else:
        str = '%ds' % (dt.tm_sec)
    return str


def dots(iterable, pre_string=None, verbose=True):
    """Adds a progress bar to a for loop.

    Keywords:
    ---------
    - `pre_string` : str
        title of the progress bar

    - `verbose` : boolean (True):
        override on whether to print at all

    """
    # print pre_string
    if verbose and (pre_string is not None):
        sys.stdout.write(pre_string)
        sys.stdout.flush()
    # initial variables
    initial_time = time.time()
    source = iterable.__iter__()
    count = 0
    max_count = max([len(iterable), 1])
    proportion = floor(10. * count / max_count)
    # run through the loop
    try:
        while True:
            n = source.next()
            yield n
            # if we have progressed by 10%, print a dot
            if not verbose:
                continue
            count += 1
            new_proportion = floor(10. * count / max_count)
            if new_proportion > proportion:
                if new_proportion <= 5:
                    sys.stdout.write('.')
                else:
                    sys.stdout.write('o')
                sys.stdout.flush()
            proportion = new_proportion
    # print time elapsed when we're done
    except StopIteration:
        if verbose:
            sys.stdout.write('[%s]\n' % time_elapsed(initial_time))
            sys.stdout.flush()
        raise StopIteration()


def numbers(iterable, key=lambda x: x.__repr__(), verbose=True, 
        header=None, pre=0, post=0):
    """Lists the progress as it goes along.

    Keywords:
    ---------
    - `key` : function
        what to print for each yield of the iterable

    - `verbose` : boolean (True)
        override on whether to print at all

    - `header` : character (None)
        make into a title, by using the provided character (e.g. '=')

    - `pre` : int (0)
        number of blank lines before

    - `post` : int (0)
        number of blank lines after

    """
    # initial variables
    source = iterable.__iter__()
    count = 0
    max_count = max([len(iterable), 1])
    proportion = floor(10. * count / max_count)
    # run through the loop
    while True:
        n = source.next()
        if verbose:
            s = '[%d/%d] %s' % (count, max_count - 1, key(n))
            if pre:
                print '\n' * (pre - 1)
            if header:
                print header * (len(s) + 1)
            print s
            if header:
                print header * (len(s) + 1)
            if post:
                print '\n' * (post - 1)
            sys.stdout.flush()
        yield n
        count += 1


names = numbers


class Timer(object):

    def __init__(self, string=None):
        self.t0 = time.time()
        if string is not None:
            sys.stdout.write(string)
            sys.stdout.flush()

    def restart(self, string=None):
        self.__init__(string=string)

    def finish(self, string=None):
        if string is None:
            string = ''
        sys.stdout.write('%s[%s]\n' % (string, time_elapsed(self.t0)))
        sys.stdout.flush()

    def next(self, string=None):
        self.finish()
        self.restart(string=string)
