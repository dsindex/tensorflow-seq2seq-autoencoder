#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tensorflow as tf
import numpy as np
import data_utils
import time
import os

class Model(object):
    def __init__(self,vocab_size,embedding_size, state_size, num_layers, num_samples, max_seq_length,max_gradient_norm, cell_type,optimizer, learning_rate, verbose, is_train):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.max_gradient_norm = max_gradient_norm
        self.cell_type = cell_type
        self.num_samples = num_samples
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.is_train = is_train
        self.global_step = tf.Variable(0, trainable=False)

        '''create encoder and decoder variables'''
        self.encoder_inputs = tf.placeholder(tf.int32, [self.max_seq_length, None]) # [max_seq_length * batch_size] tensor representing input sequences, None for variable batch_size
        self.encoder_lengths = tf.placeholder(tf.int32, [None]) # [batch_size] tensor recording each sequence's length, used by rnn cell to decide when to finish computing
        self.decoder_inputs = tf.placeholder(tf.int32, [self.max_seq_length+2,None]) # decoder_inputs add the 'GO' and 'EOS' symbol, so 2 more time steps
        self.decoder_weights = tf.placeholder(tf.float32,[self.max_seq_length+2,None]) # for the padded parts in a sequence, the weights are 0.0, which means we don't care about their loss

        '''create output projection variables'''
        # what is output projection?
        # decoder rnn output at step t (lets call it o_t)  is [state_size] dimentional; o_t*w+b is [vocab_size] dimentional, so the decoder generate words by w_t = argmax_w{o_t*w+b}
        w = tf.get_variable("proj_w", [self.state_size, self.vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable("proj_b", [self.vocab_size])
        output_projection = (w, b)
        ''' NOT USED
        # what is softmax_loss_function?
        # an in-complete softmax model which considers only [num_samples] classes to simplify loss calculation. you don't need to care about the details because the tf.nn.sampled_softmax_loss function do it automatically
        softmax_loss_function = None
        if self.num_samples>0 and self.num_samples< self.vocab_size:
            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(weights=w_t, biases = b, inputs = inputs, labels = labels, num_sampled = self.num_samples,num_classes = self.vocab_size)
            softmax_loss_function = sampled_loss
        '''

        '''create embedding and embedded inputs'''
        with tf.device("/cpu:0"): # embedding lookup only works with cpu
            embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size])
            embedded_encoder_inputs = tf.unstack(tf.nn.embedding_lookup(embedding,self.encoder_inputs)) # embedding_lookup function gets a sequence's embedded representation
            if self.verbose:
                print 'encoder_inputs', self.encoder_inputs # (30, ?)
                print 'embedded_encoder_inputs(before unstack)', tf.nn.embedding_lookup(embedding,self.encoder_inputs) # (30, ?, 128)
                print 'embedded_encoder_inputs', embedded_encoder_inputs # (?, 128), ... , (?, 128) = max_seq_length
            embedded_decoder_inputs = tf.unstack(tf.nn.embedding_lookup(embedding,self.decoder_inputs))
            if self.verbose:
                print 'decoder_inputs', self.decoder_inputs # (32, ?)
                print 'embedded_decoder_inputs(before unstack)', tf.nn.embedding_lookup(embedding,self.decoder_inputs) # (32, ?, 128)
                print 'embedded_decoder_inputs', embedded_decoder_inputs # (?, 128), ..., (?, 128) = max_seq_length + 2

        '''create rnn cell'''
        cell = tf.contrib.rnn.LSTMCell(num_units=self.state_size, state_is_tuple=True)
        if cell_type =='gru':
            cell = tf.contrib.rnn.GRUCell(self.state_size)
        if self.num_layers>1:
            cell = tf.contrib.MultiRNNCell([cell] * self.num_layers)

        '''create encoder result'''
        # here we encode the sequences to encoder_states, note that the encoder_state of a sequence is [num_layers*state_size] dimentional because it records all layers' states
        encoder_outputs, self.encoder_states = tf.contrib.rnn.static_rnn(cell=cell, inputs=embedded_encoder_inputs, sequence_length = self.encoder_lengths, dtype=tf.float32)

        '''create decoder result'''
        # weiredly, we need a loop_function here, because:
        # commonly, the seq-to-seq framework works at two modes: when training, it uses the groundtruth w_t as step-t's input
        # but when predicting, it uses a loop_function to pass the previous prediction result to current step as the input
        def loop_function(prev, _):
            prev = tf.matmul(prev,output_projection[0])+output_projection[1] # get each word's probability
            prev_symbol = tf.argmax(prev, 1) # get the most likely prediction word
            emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol) # embed the word as the next step's input
            return emb_prev
        # here we initialize the decoder_rnn with encoder_states and then try to recover the whole sequence by running the rnn
        # as it is said above, the decoder will cheat by looking into the groundtruth (only in training)
        # the decoder_outputs records each step's prediction result
        self.decoder_outputs, decoder_states = tf.contrib.legacy_seq2seq.rnn_decoder(embedded_decoder_inputs, self.encoder_states, cell,loop_function=None if self.is_train else loop_function)
        if self.verbose:
            print 'decoder_outputs(before projection)', self.decoder_outputs  # (?, 128), ..., (?, 128) = max_seq_length + 2
        self.decoder_outputs = [tf.matmul(one,output_projection[0])+output_projection[1] for one in self.decoder_outputs]
        if self.verbose:
            print 'decoder_outputs', self.decoder_outputs # (?, 4000), ..., (?, 4000) = max_seq_length + 2

        '''create loss function'''
        # as an instance, if a sequence is [GO,w1,w2,w3,EOS],then at step 0, the decoder accept 'GO', and try to predict w1, and so on... therefore decoder_truth is decoder_inputs add 1
        decoder_truth = [tf.unstack(self.decoder_inputs)[i+1] for i in xrange(self.max_seq_length+1)]
        if self.verbose:
            print 'decoder_truth', decoder_truth  # (?,), ..., (?,) == max_seq_length + 1
        # loss can by automatically cauculated with tf.nn.seq2seq.sequence_loss, and it is batch-size-normalized.
        self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decoder_outputs[:-1],decoder_truth,tf.unstack(self.decoder_weights)[:-1])
        if self.verbose:
            print 'decoder_weights', self.decoder_weights[:-1] # (31, ?), 31 == max_seq_length + 1

        '''create gradients'''
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,self.max_gradient_norm) # gradient clip is frequently used in rnn

        '''create optimizer'''
        opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        if self.optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate = self.learning_rate)
        self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        '''create saver'''
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep = 10)

    def initilize(self,model_dir,session=None):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Creating model with fresh parameters.")
            session.run(tf.global_variables_initializer())

    def get_batch(self,data_set,batch_size,random=True):
        '''get a batch of data from a data_set and do all needed preprocess
        to make them usable for the model defined above'''
        if random:
            seqs = np.random.choice(data_set,size= batch_size)
        else:
            seqs = data_set[0:batch_size]
        encoder_inputs = np.zeros((batch_size,self.max_seq_length),dtype = int)
        decoder_inputs = np.zeros((batch_size,self.max_seq_length+2),dtype = int)
        encoder_lengths = np.zeros(batch_size)
        decoder_weights = np.zeros((batch_size,self.max_seq_length+2),dtype=float)
        for i,seq in enumerate(seqs):
            encoder_inputs[i] = np.array(list(reversed(seq))+[data_utils.PAD_ID]*(self.max_seq_length-len(seq)))
            decoder_inputs[i] = np.array([data_utils.GO_ID]+seq+[data_utils.EOS_ID]+[data_utils.PAD_ID]*(self.max_seq_length-len(seq)))
            encoder_lengths[i]= len(seq)
            decoder_weights[i,0:(len(seq)+1)]=1.0
        return np.transpose(encoder_inputs), np.transpose(decoder_inputs), encoder_lengths, np.transpose(decoder_weights)

    def step(self,encoder_inputs,decoder_inputs,encoder_lengths,decoder_weights,is_train,session=None):
        '''do a uniq step of the model
        when trainning, do parameter updating; when predicting, do not.
        tranpose is necessary to get easy-to-use decoder_outputs of shape [batch_size * max_seq_length * vocab_size]'''
        feed = {}
        feed[self.encoder_inputs] = encoder_inputs
        feed[self.decoder_inputs] = decoder_inputs
        feed[self.encoder_lengths] = encoder_lengths
        feed[self.decoder_weights] = decoder_weights
        self.is_train = is_train
        if is_train:
            encoder_states,decoder_outputs,loss,_ = session.run([self.encoder_states,tf.transpose(tf.stack(self.decoder_outputs),[1,0,2]),self.loss,self.update],feed)
        else:
            encoder_states,decoder_outputs,loss = session.run([self.encoder_states,tf.transpose(tf.stack(self.decoder_outputs),[1,0,2]),self.loss],feed)
        return (encoder_states, # hidden-layer representation we are interested about
                decoder_outputs, # output
                loss) # loss

    def fit(self,train_set,validation_set,batch_size,step_per_checkpoint,max_step,model_dir):
        '''sklearn-stype fit function: use the data to fit the model'''
        with tf.Session() as session:
            self.initilize(model_dir,session)
            iteration = 0
            while True:
                iteration +=1
                start = time.time()
                # a train step
                encoder_inputs,decoder_inputs, encoder_lengths, decoder_weights= self.get_batch(train_set,batch_size)
                _, _, step_loss = self.step(encoder_inputs,decoder_inputs, encoder_lengths, decoder_weights, True ,session)
                finish = time.time()
                if self.global_step.eval() >= max_step:
                    print 'global_step have just reached max_step. done'
                    break
                print "global-step %d loss %.5f time %.2f"%(self.global_step.eval(),step_loss,finish-start)
                if self.global_step.eval()%step_per_checkpoint==0:
                    # at checkpoint we do validation and save.
                    validation_loss = 0.0
                    validation_batch_size = 10240 # lagger batch_size in validation for efficiency.
                    for i in xrange(int(np.ceil(1.0*len(validation_set)/validation_batch_size))): # constant permutation of batches in validation for consistency
                        start = i*validation_batch_size
                        end = min((i+1)*validation_batch_size,len(validation_set))
                        encoder_inputs,decoder_inputs, encoder_lengths, decoder_weights= self.get_batch(validation_set[start:end],end-start,False)
                        _, _, step_loss = self.step(encoder_inputs,decoder_inputs, encoder_lengths, decoder_weights, False ,session)
                        validation_loss+=step_loss*(end-start)/len(validation_set)
                    print "validation-loss %.5f"%(validation_loss)
                    # save
                    self.saver.save(session, os.path.join(model_dir,'checkpoint'), global_step=self.global_step)
                sys.stdout.flush()

    def inference(self,test_set,batch_size,session):
        '''inference function'''
        start = time.time()
        encoder_states_array = []
        decoder_outputs_array = []
        test_loss = 0.0
        test_batch_size = batch_size
        for i in xrange(int(np.ceil(1.0*len(test_set)/test_batch_size))):
            start = i*test_batch_size
            end = min((i+1)*test_batch_size,len(test_set))
            encoder_inputs,decoder_inputs, encoder_lengths, decoder_weights= self.get_batch(test_set[start:end],end-start,False)
            encoder_states, decoder_outputs, step_loss = self.step(encoder_inputs,decoder_inputs, encoder_lengths, decoder_weights, False ,session)
            encoder_states_array.append(encoder_states)
            decoder_outputs_array.append(decoder_outputs)
            test_loss+=step_loss*(end-start)/len(test_set)
        finish = time.time()
        if self.verbose:
            print 'duration time %.2f' % (finish-start)
        return (encoder_states_array, decoder_outputs_array, test_loss)
