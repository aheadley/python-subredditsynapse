#!/usr/bin/env python2

import random
import bz2
import json
import logging

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import numpy

from subredditsynapse.data import SEP_CHAR, CHAR_WIDTH, RedditDataDump, \
    DataSegmenter, byte2vec, vec2byte

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('ss')
logger.setLevel(logging.DEBUG)

def build_model(batch_size, segment_size, step_size, loss_rate=0.01):
    model = Sequential()
    #model.add(LSTM(segment_size + 1, input_shape=(batch_size, segment_size)))
    #model.add(Dense(1))
    #model.add(Activation('softmax'))
    model.add(LSTM(256, input_shape=(segment_size, 1, CHAR_WIDTH)))
    model.add(Dense(256))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=loss_rate))

    return model

def train_model(model, training_data):
    pass

def batch_segments(batch_size, segments):
    segs = [next(segments) for i in range(batch_size)]
    # X = numpy.asarray(

if __name__ == '__main__':
    import sys

    data_fn = sys.argv[1]

    batch_size = 10
    segment_size = 32
    step_size = 3

    segments = DataSegmenter(
        RedditDataDump(data_fn, modify_func=lambda d: bytearray(d['body'].encode('utf-8'))),
        segment_size, step_size)

    model = build_model(batch_size, segment_size, step_size)
    model.summary()

    model.fit_generator(
        segments,
        samples_per_epoch=100,
        nb_epoch=1,
        callbacks=[
            ModelCheckpoint('last-loss.h5',
                monitor='loss', verbose=0, save_best_only=False, mode='auto'),
            ModelCheckpoint('best-loss.h5',
                monitor='loss', verbose=0, save_best_only=True, mode='auto'),
            ModelCheckpoint('best-acc.h5',
                monitor='acc', verbose=0, save_best_only=True, mode='auto'),
        ]
    )

    # train_model(model, segments)
    # for i in range(10):
    #     x, y = next(segments)
    #     print '###'
    #     print '%r +> %r' % (x, y)

