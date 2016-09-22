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
from six.moves.html_parser import HTMLParser

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

SEP_CHAR    = '\x00'
CHAR_WIDTH  = 2**8

_HTML_PARSER = HTMLParser()

def RedditDataDump(fn, filter_func=lambda d: True, transform_func=lambda d: d, line_limit=None):
    logger.debug('Loading data from file: %s', fn)
    line_num = 0
    with bz2.BZ2File(fn, 'r') as f:
        while True:
            for json_line in f:
                if line_limit is not None and line_num >= line_limit:
                    break
                try:
                    datum = json.loads(json_line)
                except Exception as err:
                    logger.exception(err)
                    logger.error(err)
                    continue
                logger.debug('Loaded datum: id=% 6s len=%06d with keys: %s', datum['id'], len(json_line), datum.keys())
                if filter_func(datum):
                    logger.debug('Comment: %s', datum['body'])
                    try:
                        line_num += 1
                        yield transform_func(datum)
                    except Exception as err:
                        logger.exception(err)
                        logger.error(err)
                        continue
                else:
                    logger.debug('Skipping datum: %s', datum['body'])
            f.seek(0)
            line_num = 0

def DataSegmenter(input_data, segment_size, step_size, segment_sep=SEP_CHAR, segment_limit=None):
    while True:
        logger.debug('Starting new segment loop')
        segment_buffer = bytearray()
        segment_num = 0
        brk = False
        for datum in input_data():
            logger.debug('Segment buffer drained, filling...')
            segment_buffer += bytearray([segment_sep]) + datum
            while len(segment_buffer) > (segment_size + step_size + 1):
                seg = segment_buffer[:segment_size]
                next_char = segment_buffer[segment_size]
                segment_buffer = segment_buffer[step_size:]

                X_i = numpy.zeros((segment_size, CHAR_WIDTH), dtype=numpy.uint8)
                for c in range(segment_size):
                    X_i[c] = byte2vec(seg[c])
                y_i = byte2vec(next_char)
                segment_num += 1
                yield (X_i, y_i)
                if segment_limit is not None and segment_num >= segment_limit:
                    brk = True
                    break
            if brk:
                break


def SegmentBatcher(batch_size, segments):
    logger.debug('Creating new batcher: len=%02d', batch_size)
    while True:
        batch = [next(segments) for i in range(batch_size)]
        X_b = numpy.zeros((batch_size,) + batch[0][0].shape)
        y_b = numpy.zeros((batch_size, CHAR_WIDTH))
        for i in range(batch_size):
            X_b[i] = batch[i][0]
            y_b[i] = batch[i][1]

        yield X_b, y_b


def byte2vec(b):
    v = numpy.zeros(CHAR_WIDTH, dtype=numpy.uint8)
    v[b] = 1
    return v

def vec2byte(v):
    return numpy.argmax(v.flatten())

def comment_filter(c):
    return c['body'].strip() not in ['[removed]', '[deleted]']

def comment_transform(c):
    return bytearray(_HTML_PARSER.unescape(c['body']).encode('utf-8'))
