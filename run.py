#!/usr/bin/env python2

import random
import bz2
import json
import logging

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
import numpy

from subredditsynapse.data import SEP_CHAR, CHAR_WIDTH, RedditDataDump, \
    DataSegmenter, byte2vec, vec2byte, SegmentBatcher
from subredditsynapse.util import get_root_logger

logger = get_root_logger()

SEED_COMMENT = 'The quick brown fox jumped over the lazy dog.'

def build_model(batch_size, segment_size, step_size, layer_size=128, dropout=0.2, loss_rate=0.01):
    model = Sequential()
    model.add(LSTM(layer_size, input_shape=(segment_size, CHAR_WIDTH)))
    # model.add(Dropout(dropout))
    # model.add(LSTM(layer_size))
    # model.add(Dropout(dropout))
    model.add(Dense(CHAR_WIDTH))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=loss_rate),
        metrics=['accuracy'],
    )

    return model

def train_model(model, training_data, samples_per_epoch, nb_epoch):
    result = model.fit_generator(
        training_data,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epoch,

        validation_data= \
            SegmentBatcher(1024,
                DataSegmenter(
                    RedditDataDump('/tmp/RC_2007-08.bz2', transform_func=lambda d: bytearray(d['body'].encode('utf-8'))),
                    8, 3,
                )
            ),
        nb_val_samples=samples_per_epoch / 3,

        callbacks=[
            ModelCheckpoint('last-loss.h5',
                monitor='loss', verbose=0, save_best_only=False, mode='auto'),
            ModelCheckpoint('best-loss.h5',
                monitor='loss', verbose=0, save_best_only=True, mode='auto'),
            ModelCheckpoint('best-acc.h5',
                monitor='acc', verbose=0, save_best_only=True, mode='auto'),
        ],
    )

    return result

def prep_seed(seed, segment_size):
    seed = bytearray(seed[-(segment_size - 1):] + '\x00')
    X_p = numpy.zeros((1, segment_size, CHAR_WIDTH), dtype=numpy.bool)
    for i in range(segment_size):
        # !!! maybe needs to be a FP, see line #94
        X_p[0, i] = byte2vec(seed[i])
    logger.debug('X_p.shape=%r', X_p.shape)
    return X_p

def generate_comment(model, seed, segment_size, temperature=1.0):
    def sample_func(a):
        a = numpy.log(a) / temperature
        a = numpy.exp(a) / numpy.sum(numpy.exp(a))
        return numpy.argmax(numpy.random.multinomial(1, a, 1))
    comment = bytearray()
    seed = prep_seed(seed, segment_size)

    while True:
        prediction = model.predict(seed, verbose=0)
        # logger.debug('prediction: %r', prediction)
        # logger.debug('prediction[0]: %r', prediction[0])
        next_byte = sample_func(prediction[0])
        if next_byte == 0:
            break
        comment.append(next_byte)
        # logger.debug('Generated character: % 3d', next_byte)
        # logger.debug('Generated character: "%s" (% 3d)',
        #     unicode(bytearray(next_byte)).decode('utf-8'), next_byte)
    return unicode(comment).decode('utf-8', 'ignore')



if __name__ == '__main__':
    import sys
    import optparse

    data_fn = sys.argv[1]
    try:
        sys.argv[2]
        use_model_file = True
    except IndexError:
        use_model_file = False

    batch_size = 1024
    segment_size = 8
    step_size = 3

    samples_per_epoch = batch_size * 200
    nb_epoch = 32

    if use_model_file:
        model = load_model('model.h5')
    else:
        logger.info('Setting up data source...')
        data_src = RedditDataDump(data_fn, transform_func=lambda d: bytearray(d['body'].encode('utf-8')))
        segments = DataSegmenter(data_src, segment_size, step_size)
        batches = SegmentBatcher(batch_size, segments)

        logger.info('Building model...')
        model = build_model(batch_size, segment_size, step_size, layer_size=512)
        model.summary()

        logger.info('Training model...')
        train_model(model, batches, samples_per_epoch, nb_epoch)

        model.save('model.h5')

    logger.info('Generating a new comment...')
    c = generate_comment(model, SEED_COMMENT, segment_size, temperature=0.8)

    # print repr(c)
    logger.info('Comment: %r', c)
