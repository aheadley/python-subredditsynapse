#!/usr/bin/env python2

import random
import bz2
import json
import logging
import os.path

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy

from subredditsynapse.data import SEP_CHAR, CHAR_WIDTH, RedditDataDump, \
    DataSegmenter, byte2vec, vec2byte, SegmentBatcher, comment_transform, \
    comment_filter
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

def train_model(model, training_data, validation_data, samples_per_epoch, nb_epoch):
    result = model.fit_generator(
        training_data,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epoch,

        validation_data=validation_data,
        nb_val_samples=samples_per_epoch / 3,

        callbacks=[
            ModelCheckpoint('model.best-acc.h5',
                monitor='acc', verbose=0, save_best_only=True, mode='auto'),
            EarlyStopping(monitor='val_acc', patience=2),
            EarlyStopping(monitor='acc', patience=3),
        ],
    )

    return result

def prep_seed(seed, segment_size):
    seed = bytearray(seed[-(segment_size - 1):] + '\x00')
    X_p = numpy.zeros((1, segment_size, CHAR_WIDTH), dtype=numpy.bool)
    for i in range(segment_size):
        X_p[0, i] = byte2vec(seed[i])
    return X_p

def generate_comment(model, seed, segment_size, temperature=1.0):
    def sample_func(a):
        a = a.astype('float64')
        a = numpy.log(a) / temperature
        a = numpy.exp(a) / numpy.sum(numpy.exp(a))
        try:
            v = numpy.random.multinomial(1, a)
        except ValueError:
            v = numpy.random.multinomial(1, a/2)
        return numpy.argmax(v)
    comment = bytearray()
    buf = prep_seed(seed, segment_size)

    while True:
        pred = model.predict(buf, verbose=0)[0]
        next_byte = sample_func(pred)
        if next_byte == 0:
            break
        comment.append(next_byte)
        buf[:-1] = buf[1:]
        buf[-1] = byte2vec(next_byte)

    return comment.decode('latin-1')

if __name__ == '__main__':
    import sys
    import optparse

    parser = optparse.OptionParser()

    parser.add_option('-T', '--train',
        action='store_true', default=False)
    parser.add_option('-b', '--batch-size',
        type='int', default=1024)
    parser.add_option('-g', '--segment-size',
        type='int', default=16)
    parser.add_option('-S', '--step-size',
        type='int', default=1)
    parser.add_option('-l', '--layer-size',
        type='int', default=128)
    parser.add_option('-s', '--samples-per-epoch',
        type='int', default=10)
    parser.add_option('-e', '--epochs',
        type='int', default=1)
    parser.add_option('-V', '--validation-data')

    parser.add_option('-P', '--predict',
        action='store_true', default=False)
    parser.add_option('-t', '--temperature',
        type='float', default=0.8)

    parser.add_option('-f', '--model-file',
        default=None)


    parser.add_option('-v', '--verbose',
        action='store_true', default=False)

    opts, args = parser.parse_args()

    if opts.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if opts.model_file is not None and os.path.exists(opts.model_file):
        logger.info('Loading model from file: %s', opts.model_file)
        model = load_model(opts.model_file)
    else:
        logger.info('Building model...')
        model = build_model(opts.batch_size, opts.segment_size, opts.step_size,
            layer_size=opts.layer_size)
    if opts.verbose:
        model.summary()

    if opts.train:
        samples_per_epoch = opts.samples_per_epoch * opts.batch_size
        if opts.validation_data:
            validation_data = SegmentBatcher(opts.batch_size,
                DataSegmenter(
                    lambda: RedditDataDump(opts.validation_data,
                        transform_func=comment_transform, filter_func=comment_filter),
                    opts.segment_size, opts.step_size,
                )
            )
        else:
            validation_data = None

        for src_data_fn in args:
            logger.info('Loading data source: %s', src_data_fn)
            data_src = lambda: RedditDataDump(src_data_fn,
                transform_func=comment_transform, filter_func=comment_filter)
            segments = DataSegmenter(data_src, opts.segment_size, opts.step_size,
                segment_limit=opts.batch_size*opts.samples_per_epoch)
            training_data = SegmentBatcher(opts.batch_size, segments)
            logger.info('Training model on %d batches * %d epochs...',
                opts.samples_per_epoch, opts.epochs)
            train_model(model, training_data, validation_data, samples_per_epoch, opts.epochs)

            if opts.model_file is not None:
                logger.info('Saving model...')
                model.save(opts.model_file)

    if opts.predict:
        logger.info('Generating a new comment...')
        comment = generate_comment(model, SEED_COMMENT, opts.segment_size, temperature=0.8)

        logger.info('Generated comment: %s', comment)
