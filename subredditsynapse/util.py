# -*- coding: utf-8 -*-

# Copyright (C) 2013  Alex Headley  <aheadley@waysaboutstuff.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import logging


try:
    NullHandler = logging.NullHandler
except AttributeError:
    class NullHandler(logging.Handler):
        level = 100

        def handle(self, record):
            pass

        def emit(self, record):
            pass

        def createLock(self):
            self.lock = None

LOG_FORMAT = logging.Formatter('[%(asctime)s] %(levelname)8s - %(name)s: %(message)s')

def get_root_logger(level=logging.DEBUG):
    l = logging.getLogger()
    l.setLevel(level)
    # l.addHandler(NullHandler)
    stderr = logging.StreamHandler()
    stderr.setFormatter(LOG_FORMAT)
    l.addHandler(stderr)
    return l
