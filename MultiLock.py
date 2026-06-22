"""File-based advisory lock for safe concurrent NetCDF database access.

Wraps ``simpleflock`` with an additional per-process lock file layer so that
multiple worker processes can safely read and write the same NetCDF database
without data corruption.
"""
import os
import glob
import time
import simpleflock
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import config

class MultiLock():
    """Context manager that provides exclusive access to a file path.

    Usage::

        with MultiLock('/path/to/database.nc'):
            ds = xr.open_dataset('/path/to/database.nc')
            # modify ds ...
            ds.to_netcdf('/path/to/database.nc')
    """

    def __init__(self, path):
        self._path = path

    def __enter__(self):
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

        os.remove(self._this_lockfile)