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
import bz2
import json

import numpy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SEP_CHAR    = '\x00'
CHAR_WIDTH  = 2**8

def RedditDataDump(fn, filter_func=lambda d: True, modify_func=lambda d: d):
    logger.debug('Loading data from file: %s', fn)
    with bz2.BZ2File(fn, 'r') as f:
        for json_line in f:
            try:
                datum = json.loads(json_line)
            except Exception as err:
                logger.exception(err)
                logger.error(err)
                continue
            logger.debug('Loaded datum: id=% 6s len=%06d with keys: %s', datum['id'], len(json_line), datum.keys())

            if filter_func(datum):
                try:
                    yield modify_func(datum)
                    # yield bytearray(data['body'].encode('utf-8'))
                except Exception as err:
                    logger.exception(err)
                    logger.error(err)
                    continue

def DataSegmenter(input_data, segment_size, step_size, segment_sep=SEP_CHAR):
    segment_buffer = bytearray()

    for datum in input_data:
        segment_buffer += bytearray([segment_sep]) + datum
        while len(segment_buffer) > (segment_size + step_size + 1):
            seg = segment_buffer[:segment_size]
            next_char = segment_buffer[segment_size]
            segment_buffer = segment_buffer[step_size:]

            X_i = numpy.zeros((segment_size, CHAR_WIDTH))
            for c in range(segment_size):
                X_i[c][0] = byte2vec(seg[c])
            y_i = byte2vec(next_char)
            yield (X_i, y_i)
        logger.debug('Segment buffer drained, filling...')

def byte2vec(b):
    v = numpy.zeros(CHAR_WIDTH)
    v[b] = 1
    return v

def vec2byte(v):
    return numpy.argmax(v.flatten())
