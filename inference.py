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
import time

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--verbose', help='verbose mode')
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
    args = parser.parse_args()

    '''read vocab'''
    dictionary, reverse_dictionary = data_utils.initialize_vocabulary(args.vocab_path)
    
    '''create model'''
    is_train = False
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
                                            args.verbose,
                                            is_train)

    '''restore model and do inference'''
    if not gfile.Exists(args.model_path):
        raise ValueError('Model dir path %s not found.', args.model_path)
    with tf.Session() as session:
        # restore model
        model.initilize(args.model_path, session)
        number_of_sent = 0
        while 1:
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                break
            if not line:
                break
            line = line.strip()
            if not line : continue
            start_time = time.time()
            print 'line:'
            print line
            # prepare data
            test_set = []
            ids = data_utils.sentence_to_token_ids(line, dictionary)
            if len(ids) > args.max_seq_length:
              ids = ids[:args.max_seq_length]
            test_set.append(ids)
            print 'test_set:'
            print test_set
            batch_size = 1
            # do inference
            encoder_states_array, decoder_outputs_array, test_loss = model.inference(test_set, batch_size, session)
            encoder_states = encoder_states_array[0]
            decoder_outputs = decoder_outputs_array[0]
            print 'encoder_states'
            print encoder_states
            print 'decoder_outputs'
            print decoder_outputs[0]
            out = []
            for out_vec in decoder_outputs_array[0][0]: # batch_size = 1
                out_idx = np.argmax(out_vec)
                out.append(out_idx)
            print out
            print 'out to sentence:'
            print ' '.join(data_utils.token_ids_to_sentence(out, reverse_dictionary))
            duration_time = time.time() - start_time
            s = 'duration_time = ' + str(duration_time) + ' sec'
            sys.stderr.write(s + '\n')
            number_of_sent += 1
        sys.stderr.write("number_of_sent = %d\n" % (number_of_sent))


if __name__ == '__main__':
    main()
