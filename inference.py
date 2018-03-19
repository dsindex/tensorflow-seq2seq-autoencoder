#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import argparse
import numpy as np
import tensorflow as tf
import data_utils
from tensorflow.python.platform import gfile
import seq2seq_autoencoder_model
import os

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-path',help='path to saved model')
    arg('--vocab-path',help='path to saved vocab')
    arg('--max-seq-length', type=int, default=30,help='the longest sequence length supported by the model, trainning data longer than this will be omitted, testing data longer than this will cause error')
    arg('--vocab-size', type=int, default=4000,help='how many most frequent words to keep, words beyond the vocabulary will be labeled as UNK')
    arg('--embedding-size', type=int, default=128,help='size of word embedding')
    arg('--state-size', type=int, default=128,help = 'size of hidden states')
    arg('--num-layers', type=int, default=1, help='number of hidden layers')
    arg('--cell', default='gru', help='cell type: lstm, gru')
    arg('--num-samples', type=int, default=256, help = 'number of sampled softmax')
    arg('--max-gradient-norm', type=float, default=5.0, help='gradient norm is commonly set as 5.0 or 15.0')
    arg('--optimizer',default='adam', help='Optimizer: adam, adadelta')
    arg('--learning-rate',type=float, default=0.01)
    arg('--batch-size', type=int, default=64)
    arg('--test-path',help='path to test data file')
    args = parser.parse_args()

    is_train = False

    '''read vocab'''
    dictionary, reverse_dictionary = data_utils.initialize_vocabulary(args.vocab_path)

    '''prepare test data'''
    test_set = []
    with gfile.GFile(args.test_path, mode='rb') as test_file:
        for line in test_file:
            ids = data_utils.sentence_to_token_ids(line, dictionary)
            test_set.append(ids)
    test_set = [one for one in test_set if len(one)<=args.max_seq_length] 
    
    '''create model'''
    model = seq2seq_autoencoder_model.Model(args.vocab_size,
                                            args.embedding_size,
                                            args.state_size,
                                            args.num_layers,
                                            args.num_samples,
                                            args.max_seq_length,
                                            args.max_gradient_norm,
                                            args.cell,
                                            args.optimizer,
                                            args.learning_rate,
                                            is_train)

    '''restore model and inference'''
    if not gfile.Exists(args.model_path):
        raise ValueError('Model dir path %s not found.', args.model_path)
    with tf.Session() as session:
        model.initilize(args.model_path, session)
        model.inference(test_set, args.batch_size, session)

if __name__ == '__main__':
    main()
