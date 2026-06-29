"""File-based advisory lock for safe concurrent NetCDF database access.

Wraps ``simpleflock`` with an additional per-process lock file layer so that
multiple worker processes can safely read and write the same NetCDF database
without data corruption.
"""
import glob
import logging
import os
import time

import numpy as np
import simpleflock

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MultiLock():
    """Context manager that provides exclusive access to a file path.

    Layers a per-process ``.pid.lock`` sentinel file on top of
    ``simpleflock`` to close the race window where ``simpleflock`` can
    briefly grant the system lock to two processes simultaneously.  The
    calling process:

    1. Acquires the ``simpleflock`` system lock on ``<path>.lock``.
    2. Creates its own ``<path>.<pid>.lock`` sentinel file.
    3. Releases the system lock.
    4. Loops until no *other* sentinel files exist, sleeping a random
       sub-second interval on each iteration to break ties.

    This guarantees that at most one process holds the advisory lock at any
    time, even when the underlying ``simpleflock`` implementation has a
    brief grant-to-two bug.

    Parameters
    ----------
    path : str or os.PathLike
        Absolute path to the file being protected (typically a NetCDF
        database such as ``modal_accel.nc``).

    Examples
    --------
    ::

        with MultiLock('/path/to/database.nc'):
            ds = xr.open_dataset('/path/to/database.nc')
            # modify ds …
            ds.to_netcdf('/path/to/database.nc')
    """

    def __init__(self, path):
        """Store the path of the resource to protect.

        Parameters
        ----------
        path : str or os.PathLike
            Absolute path to the file that will be locked.
        """
        self._path = path
        self._this_lockfile = None

    def __enter__(self):
        """Acquire exclusive access to the protected file.

        Blocks until no other process holds the lock.  Uses a randomised
        back-off sleep (uniform in [0, 1) seconds) to avoid live-lock when
        two processes race for the same resource.

        Returns
        -------
        None
            This context manager yields nothing; callers use ``with
            MultiLock(path):`` without ``as``.
        """
        self._this_lockfile = f'{self._path}.{config.pid}.lock'

        # simpleflock sometimes gives lock to two processes
        with simpleflock.SimpleFlock(f"{self._path}.lock"):
            while True:
                lockfile_list = glob.glob(f'{self._path}.*.lock')
                # print(lockfile_list)
                if len(lockfile_list) > 0:
                    if len(
                            lockfile_list) == 1 and lockfile_list[0] == self._this_lockfile:
                        # this processes lockfile is the only one, we can
                        # continue to modify the ds safely
                        logger.debug(f'Acquired lock on {self._path}.lock')
                        return
                    elif self._this_lockfile in lockfile_list:
                        # another process has created a lockfile meanwhile ->
                        # start over
                        os.remove(self._this_lockfile)
                        time.sleep(np.random.random())
                    else:
                        # another process currently holds the lock for this
                        # file
                        logger.warning(
                            'Wating for lockfile to release: {}'.format(lockfile_list))
                        time.sleep(np.random.random())
                else:
                    # if no other lockfile exists -> create one
                    # continue in while loop to check for race conditions with
                    # othe processes
                    _fd = open(self._this_lockfile, 'w+')
                    _fd.close()

    def __exit__(self, *args):
        """Release the lock by removing this process's sentinel file.

        Parameters
        ----------
        *args : tuple
            Exception info (``exc_type``, ``exc_val``, ``exc_tb``); ignored.
            The lock is released unconditionally even when the body of the
            ``with`` block raises an exception.
        """
        os.remove(self._this_lockfile)
