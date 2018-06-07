# Future import
from __future__ import absolute_import, division, print_function

# Standard imports
import sys
import os
import subprocess
import re
from re import compile as _Re
import random
import csv
import dill as pickle
import gzip
import math
from math import exp, log
import random
from copy import deepcopy
from builtins import range
from collections import OrderedDict
import numpy as np
import pandas as pd

# RDKit imports
import rdkit
from rdkit import rdBase
import rdkit.Chem.AllChem as Chem
from rdkit.Chem import Crippen, MolFromSmiles, MolToSmiles, Descriptors

# PyMatGen imports
import pymatgen as mg 
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

#####################################################
# This built from the ORGANIC program of Aspuru-Guzik
# It has been only lightly modified by S. Ryno to 
# increase generality and compatibility with more
# modern packages
#####################################################

###############
# GPU Utilities
###############
def run_command(cmd):
    """
    Run terminal command and return output as string
    """
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode('ascii')

def list_available_gpus():
    """
    Returns a list of the available GPU IDs and models
    """
    output = run_command("nvidia-smi -L")
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldn't parse " + line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """
    Returns mapping of GPU IDs to currently allocated memory
    """
    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """
    Returns the GPU ID with the current lowest allocated memory\
    """
    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu

# ML imports
try:
    gpu_free_number = str(pick_gpu_lowest_memory())
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_free_number)
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    from keras import backend as K
except Exception:
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras_tqdm import TQDMCallback
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow import logging
from tqdm import tqdm

#############################################
# NN Metrics
# Class to handle Keras neural network models
#############################################
class KerasNN(object):
    """
    Class for handling Keras neural network models
    """
    def __init__(self, label, nBits=4096):
        """
        Initializes Keras model

        Arguments
        --------------
            - label: Identifies the property to be 
                     predicted by the neural network.
            - nBits: Refers to the number of bits in 
                     which the Morgan fingerprints 
                     are encoded. Defaults=4096
        """
        self.label = label
        self.graph = tf.Graph()
        self.nBits = nBits

    def predict(self, smiles, batch_size=100):
        """
        Computes the predictions for a batch of molecules

        Arguments
        -----------
            - smiles: Array or list containing the
            SMILES representation of the molecules
            - batch_size: Optional. Size of the batch
            used for computing the properties

        Returns
        -----------
            A list containing the predictions
        """
        with self.graph.as_default():
            input_x = self.computeFingerprints(smiles)
            return self.model.predict(input_x, batch_size=batch_size)

    def evaluate(self, train_x, train_y):
        """
        Evaluates the accuracy of the method

        Arguments
        -----------
            - train_x: Array or list containing the
            SMILES representation of the molecules
            - train_y: The real values of the desired
            properties

        Returns
        -----------
            Test loss
        """

        with self.graph.as_default():
            input_x = self.computeFingerprints(train_x)
            return self.model.evaluate(input_x, train_y, verbose=0)

    def load(self, file):
        """
        Loads a previously trained model

        Arguments
        -----------
            - file: A string pointing to the .h5 file
        """

        with self.graph.as_default():
            self.model = load_model(file)

    def train(self, train_x, train_y, batch_size, nepochs, earlystopping=True, min_delta=0.001):
        """
        Trains the model
        The model is saved in the current directory under the name "label".h5

        Arguments
        -----------
            - train_x: Array or list containing the
               SMILES representation of the molecules
            - train_y: The real values of the desired
               properties
            - batch_size: The size of the batch
            - nepochs: The maximum number of epochs
            - earlystopping: Boolean specifying whether early
            stopping will be used or not. Default=True
            - min_delta: If earlystopping is True, the variation
            on the validation set's value which will trigger
            the stopping
        """
        with self.graph.as_default():
            """
            Use a Sequential model with dropout rate of 0.2, 2 hidden
            layers, and 1 output layer. All layers are full-connected
            """
            self.model = Sequential()
            self.model.add(Dropout(0.2, input_shape=(self.nBits,)))
            self.model.add(BatchNormalization())
            self.model.add(Dense(300, activation='relu', kernel_initializer='normal'))
            self.model.add(Dense(300, activation='relu', kernel_initializer='normal'))
            self.model.add(Dense(1, activation='linear', kernel_initializer='normal'))
            self.model.compile(optimizer='adam', loss='mse')

            input_x = self.computeFingerprints(train_x)

            if earlystopping is True:
                callbacks = [EarlyStopping(monitor='val_loss',
                                           min_delta=min_delta,
                                           patience=10,
                                           verbose=0,
                                           mode='auto'),
                             TQDMCallback()]
            else:
                callbacks = [TQDMCallback()]

            self.model.fit(input_x, train_y,
                           shuffle=True,
                           epochs=nepochs,
                           batch_size=batch_size,
                           validation_split=0.1,
                           verbose=2,
                           callbacks=callbacks)

            self.model.save('{}.h5'.format(self.label))

    def computeFingerprints(self, smiles):
        """
        Computes Morgan fingerprints using RDKit

        Arguments
        -----------
            - smiles: An array or list of molecules in
               the SMILES codification

        Returns
        -----------
            A numpy array containing Morgan fingerprints
            bitvectors
        """
        if isinstance(smiles, str):     # We need smiles as lists even if 1 smile
            smiles = [smiles]

        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        fps = [Chem.GetMorganFingerprintAsBitVect(mol, 12, nBits=self.nBits) for mol in mols]
        bitvectors = [self.fingerprintToBitVect(fp) for fp in fps]
        return np.asarray(bitvectors)

    def fingerprintToBitVect(self, fp):
        """
        Transforms a Morgan fingerprint to a bit vector

        Arguments
        -----------
            - fp: Morgan fingerprint

        Returns
        -----------
            A bit vector
        """
        return np.asarray([float(i) for i in fp])

#################
# Generator model
#################
class Generator(object):
    """
    Class for the generative model
    """
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, 
                 sequence_length, start_token, learning_rate=0.001,
                 reward_gamma=0.95, temperature=1.0, grad_clip=5.0):
        """
        Sets parameters and defines the model architecture
        """

        """
        Set specific parameters
        """
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.reward_gamma = reward_gamma
        self.temperature = temperature
        self.grad_clip = grad_clip
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)

        """
        Set important internal variables
        """
        self.g_params = []  # This list will be updated with LSTM's parameters
        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))
        self.x = tf.placeholder(  # true data, not including start token
            tf.int32, shape=[self.batch_size, self.sequence_length])
        self.rewards = tf.placeholder(  # rom rollout policy and discriminator
            tf.float32, shape=[self.batch_size, self.sequence_length])

        """
        Define generative model
        """
        with tf.variable_scope('generator'):
            self.g_embeddings = tf.Variable(
                self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            self.g_recurrent_unit = self.create_recurrent_unit(
                self.g_params)  # maps h_tm1 to h_t for generator
            self.g_output_unit = self.create_output_unit(
                self.g_params)  # maps h_t to o_t (output token logits)

        """
        Process the batches
        """
        with tf.device("/cpu:0"):
        # with tf.device("/device:GPU:0"):
            inputs = tf.split(axis=1, num_or_size_splits=self.sequence_length,
                              value=tf.nn.embedding_lookup(self.g_embeddings,
                                                           self.x))
            self.processed_x = tf.stack(  # seq_length x batch_size x emb_dim
                [tf.squeeze(input_, [1]) for input_ in inputs])
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        """
        Generative process
        """
        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32,
                                             size=self.sequence_length,
                                             dynamic_size=False,
                                             infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32,
                                             size=self.sequence_length,
                                             dynamic_size=False,
                                             infer_shape=True)

        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(
                log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(
                self.g_embeddings, next_token)  # batch x emb_dim
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_emb, 1.0, 0.0),
                                                             tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o, gen_x

        _, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
                                            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
                                            body=_g_recurrence,
                                            loop_vars=(tf.constant(0, dtype=tf.int32),
                                                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                                                       self.h0, gen_o, gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0]) # batch_size x seq_length

        """
        Pretraining
        """
        g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                     dynamic_size=False, infer_shape=True)

        g_logits = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions, g_logits):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            g_predictions = g_predictions.write(
                i, tf.nn.softmax(o_t))  # batch x vocab_size
            g_logits = g_logits.write(i, o_t)  # batch x vocab_size
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions, g_logits

        _, _, _, self.g_predictions, self.g_logits = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),tf.nn.embedding_lookup(
                                                        self.g_embeddings, self.start_token),
                                    self.h0, g_predictions, g_logits))

        self.g_predictions = tf.transpose(
            self.g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        self.g_logits = tf.transpose(
            self.g_logits.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size
        self.pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0))) / (self.sequence_length * self.batch_size)

        pretrain_opt = self.g_optimizer(self.learning_rate)  # training updates
        self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.g_params), 
                                                        self.grad_clip)
        self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))

        """
        Unsupervised Training
        """
        self.g_loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)), 1) 
                                        * tf.reshape(self.rewards, [-1]))
        g_opt = self.g_optimizer(self.learning_rate)
        self.g_grad, _ = tf.clip_by_global_norm(
            tf.gradients(self.g_loss, self.g_params), self.grad_clip)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))

    def generate(self, session):
        """Generates a batch of samples"""
        outputs = session.run([self.gen_x])
        return outputs[0]

    def pretrain_step(self, session, x):
        """Performs a pretraining step on the generator"""
        outputs = session.run([self.pretrain_updates, self.pretrain_loss,
                               self.g_predictions], feed_dict={self.x: x})
        return outputs

    def generator_step(self, sess, samples, rewards):
        """Performs a training step on the generator"""
        feed = {self.x: samples, self.rewards: rewards}
        _, g_loss = sess.run([self.g_updates, self.g_loss], feed_dict=feed)
        return g_loss

    def init_matrix(self, shape):
        """Returns a normally initialized matrix of a given shape"""
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        """Returns a vector of zeros of a given shape"""
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        """Defines the recurrent process in the LSTM"""

        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix(
            [self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        """Defines the output part of the LSTM."""

        self.Wo = tf.Variable(self.init_matrix(
            [self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def g_optimizer(self, *args, **kwargs):
        """Sets the optimizer."""
        return tf.train.AdamOptimizer(*args, **kwargs)

#################
# Rollout model
#################
class Rollout(object):
    """
    Class for the Rollout policy model
    """
    def __init__(self, lstm, update_rate, pad_num):
        """
        Sets parameters and defines the model architecture
        """
        self.lstm = lstm
        self.update_rate = update_rate
        self.pad_num = pad_num
        self.num_emb = self.lstm.num_emb
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.sequence_length = self.lstm.sequence_length
        self.start_token = tf.identity(self.lstm.start_token)
        self.learning_rate = self.lstm.learning_rate

        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        # maps h_tm1 to h_t for generator
        self.g_recurrent_unit = self.create_recurrent_unit()
        # maps h_t to o_t (output token logits)
        self.g_output_unit = self.create_output_unit()

        ##############################
        # start placeholder definition
        ##############################
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
        self.given_num = tf.placeholder(tf.int32)
        # sequence of indices of generated data generated by generator, not
        # including start token

        # processed for batch
        with tf.device("/cpu:0"):
        # with tf.device("/device:GPU:0"):
            inputs = tf.split(axis=1, num_or_size_splits=self.sequence_length,
                              value=tf.nn.embedding_lookup(self.g_embeddings, self.x))
            self.processed_x = tf.stack(
                [tf.squeeze(input_, [1]) for input_ in inputs])  # seq_length x batch_size x emb_dim

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        ta_x = tensor_array_ops.TensorArray(
            dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(tf.transpose(self.x, perm=[1, 0]))
        ##############################
        # end placeholder definition
        ##############################

        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            x_tp1 = ta_emb_x.read(i)
            gen_x = gen_x.write(i, ta_x.read(i))
            return i + 1, x_tp1, h_t, given_num, gen_x

        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(
                log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(
                self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, given_num, gen_x

        i, x_t, h_tm1, given_num, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, self.given_num, gen_x))

        _, _, _, _, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, self.gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        # batch_size x seq_length
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])

    def get_reward(self, sess, input_x, rollout_num, cnn, reward_fn=None, D_weight=1):
        """
        Calculates the rewards for a list of SMILES strings
        """
        reward_weight = 1 - D_weight
        rewards = []
        for i in range(rollout_num):

            already = []
            for given_num in range(1, self.sequence_length):
                feed = {self.x: input_x, self.given_num: given_num}
                outputs = sess.run([self.gen_x], feed)
                generated_seqs = outputs[0]  # batch_size x seq_length
                gind = np.array(range(len(generated_seqs)))

                feed = {cnn.input_x: generated_seqs,
                        cnn.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(cnn.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])

                if reward_fn:

                    ypred = D_weight * ypred
                    # Delete sequences that are already finished,
                    # and add their rewards
                    for k, r in reversed(already):
                        generated_seqs = np.delete(generated_seqs, k, 0)
                        gind = np.delete(gind, k, 0)
                        ypred[k] += reward_weight * r

                    # If there are still seqs, calculate rewards
                    if generated_seqs.size:
                        rew = reward_fn(generated_seqs)

                    # Add the just calculated rewards
                    for k, r in zip(gind, rew):
                        ypred[k] += reward_weight * r

                    # Choose the seqs finished in the last iteration
                    for j, k in enumerate(gind):
                        if input_x[k][given_num] == self.pad_num and input_x[k][given_num-1] == self.pad_num:
                            already.append((k, rew[j]))
                    already = sorted(already, key=lambda el: el[0])

                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred       

            # Last char reward
            feed = {cnn.input_x: input_x, cnn.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(cnn.ypred_for_auc, feed)
            if reward_fn:
                ypred = D_weight * np.array([item[1]
                                             for item in ypred_for_auc])
                ypred += reward_weight * reward_fn(input_x)
            else:
                ypred = np.array([item[1] for item in ypred_for_auc])

            if i == 0:
                rewards.append(ypred)
            else:
                rewards[-1] += ypred

        rewards = np.transpose(np.array(rewards)) / \
            (1.0 * rollout_num)  # batch_size x seq_length
        return rewards

    def create_recurrent_unit(self):
        """Defines the recurrent process in the LSTM"""

        # Weights and Bias for input and hidden tensor
        self.Wi = tf.identity(self.lstm.Wi)
        self.Ui = tf.identity(self.lstm.Ui)
        self.bi = tf.identity(self.lstm.bi)

        self.Wf = tf.identity(self.lstm.Wf)
        self.Uf = tf.identity(self.lstm.Uf)
        self.bf = tf.identity(self.lstm.bf)

        self.Wog = tf.identity(self.lstm.Wog)
        self.Uog = tf.identity(self.lstm.Uog)
        self.bog = tf.identity(self.lstm.bog)

        self.Wc = tf.identity(self.lstm.Wc)
        self.Uc = tf.identity(self.lstm.Uc)
        self.bc = tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def update_recurrent_unit(self):
        """
        Updates the weights and biases of the rollout's LSTM
        recurrent unit following the results of the training
        """

        # Weights and Bias for input and hidden tensor
        self.Wi = self.update_rate * self.Wi + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wi)
        self.Ui = self.update_rate * self.Ui + \
            (1 - self.update_rate) * tf.identity(self.lstm.Ui)
        self.bi = self.update_rate * self.bi + \
            (1 - self.update_rate) * tf.identity(self.lstm.bi)

        self.Wf = self.update_rate * self.Wf + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wf)
        self.Uf = self.update_rate * self.Uf + \
            (1 - self.update_rate) * tf.identity(self.lstm.Uf)
        self.bf = self.update_rate * self.bf + \
            (1 - self.update_rate) * tf.identity(self.lstm.bf)

        self.Wog = self.update_rate * self.Wog + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wog)
        self.Uog = self.update_rate * self.Uog + \
            (1 - self.update_rate) * tf.identity(self.lstm.Uog)
        self.bog = self.update_rate * self.bog + \
            (1 - self.update_rate) * tf.identity(self.lstm.bog)

        self.Wc = self.update_rate * self.Wc + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wc)
        self.Uc = self.update_rate * self.Uc + \
            (1 - self.update_rate) * tf.identity(self.lstm.Uc)
        self.bc = self.update_rate * self.bc + \
            (1 - self.update_rate) * tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self):
        """
        Defines the output process in the LSTM
        """

        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_output_unit(self):
        """
        Updates the weights and biases of the rollout's LSTM
        output unit following the results of the training
        """

        self.Wo = self.update_rate * self.Wo + \
            (1 - self.update_rate) * tf.identity(self.lstm.Wo)
        self.bo = self.update_rate * self.bo + \
            (1 - self.update_rate) * tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_params(self):
        """
        Updates all parameters in the rollout's LSTM
        """
        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.update_recurrent_unit()
        self.g_output_unit = self.update_output_unit()

################################
# Mol Methods
# Methods for SMILES parsing
# and molecular metrics handling
################################

#########
# DATA IO
#########
def read_smi(filename):
    """
    Reads SMILES from a .smi file

    Arguments
    -----------
        - filename: String pointing to the .smi file

    Returns
    -----------
        - List of SMILES strings
    """
    with open(filename) as file:
        smiles = file.readlines()
    smiles = [i.strip() for i in smiles]
    return smiles

def read_smiles_csv(filename):
    """
    Reads SMILES from a .csv file

    Arguments
    -----------
        - filename: String pointing to the .csv file

    Returns
    -----------
        - List of SMILES strings

    Note
    -----------
        This function will assume that the SMILES are
        in column 0.
    """
    with open(filename) as file:
        reader = csv.reader(file)
        smiles_idx = next(reader).index("smiles")
        data = [row[smiles_idx] for row in reader]
    return data

def load_train_data(filename):
    """
    Loads training data from a .csv or .smi file

    Arguments
    -----------
        - filename: String pointing to the .csv or .smi file
    """
    ext = filename.split(".")[-1]
    if ext == 'csv':
        return read_smiles_csv(filename)
    if ext == 'smi':
        return read_smi(filename)
    else:
        raise ValueError('data is not smi or csv!')
    return

def save_smi(name, smiles):
    """
    Saves SMILES data as a .smi file

    Arguments
    -----------
        - filename: String pointing to the .smi file
        - smiles: List of SMILES strings to be saved
    """
    if not os.path.exists('epoch_data'):
        os.makedirs('epoch_data')
    smi_file = os.path.join('epoch_data', "{}.smi".format(name))
    with open(smi_file, 'w') as afile:
        afile.write('\n'.join(smiles))
    return

########################
# Mathematical Utilities
########################
def checkarray(x):
    """
    Checks if data is an array and not a single value
    """
    if type(x) == np.ndarray or type(x) == list:
        if x.size == 1:
            return False
        else:
            return True
    else:
        return False

def gauss_remap(x, x_mean, x_std):
    """
    Remaps a given value to a gaussian distribution

    Arguments
    -----------
        - x: Value to be remapped
        - x_mean: Mean of the distribution
        - x_std: Standard deviation of the distribution
    """
    return np.exp(-(x - x_mean)**2 / (x_std**2))

def remap(x, x_min, x_max):
    """
    Remaps a given value to [0, 1]

    Arguments
    -----------
        - x: Value to be remapped
        - x_min: Minimum value (will correspond to 0)
        - x_max: Maximum value (will correspond to 1)

    Note
    -----------
        If x > x_max or x < x_min, the value will be outside
        of the [0, 1] interval
    """
    if x_max != 0 and x_min != 0:
        return 0
    elif x_max - x_min == 0:
        return x
    else:
        return (x - x_min) / (x_max - x_min)

def constant_range(x, x_low, x_high):
    """
    Checks is data is in a given range
    """
    if checkarray(x):
        return np.array([constant_range_func(xi, x_low, x_high) for xi in x])
    else:
        return constant_range_func(x, x_low, x_high)

def constant_range_func(x, x_low, x_high):
    """
    Returns 1 if x is in [x_low, x_high] and 0 if not
    """
    if x <= x_low or x >= x_high:
        return 0
    else:
        return 1

def constant_bump_func(x, x_low, x_high, decay=0.025):
    if x <= x_low:
        return np.exp(-(x - x_low)**2 / decay)
    elif x >= x_high:
        return np.exp(-(x - x_high)**2 / decay)
    else:
        return 1

def constant_bump(x, x_low, x_high, decay=0.025):
    if checkarray(x):
        return np.array([constant_bump_func(xi, x_low, x_high, decay) for xi in x])
    else:
        return constant_bump_func(x, x_low, x_high, decay)

def smooth_plateau(x, x_point, decay=0.025, increase=True):
    if checkarray(x):
        return np.array([smooth_plateau_func(xi, x_point, decay, increase) for xi in x])
    else:
        return smooth_plateau_func(x, x_point, decay, increase)

def smooth_plateau_func(x, x_point, decay=0.025, increase=True):
    if increase:
        if x <= x_point:
            return np.exp(-(x - x_point)**2 / decay)
        else:
            return 1
    else:
        if x >= x_point:
            return np.exp(-(x - x_point)**2 / decay)
        else:
            return 1

def pct(a, b):
    if len(b) == 0:
        return 0
    return float(len(a)) / len(b)

def rectification(x, x_low, x_high, reverse=False):

    if checkarray(x):
        return np.array([rec_fun(xi, x_low, x_high, reverse) for xi in x])
    else:
        return rec_fun(x, x_low, x_high, reverse)

def rec_fun(x, x_low, x_high, reverse=False):
    if reverse == True:
        if x_low <= x <= x_high:
            return 0
        else:
            return x
    else:
        if x_low <= x <= x_high:
            return x
        else:
            return 0

def asym_rectification(x, y, reverse=False):

    if checkarray(x):
        return np.array([asymrec_fun(xi, y, reverse=reverse) for xi in x])
    else:
        return asymrec_fun(x, y, reverse=reverse)

def asymrec_fun(x, y, reverse=False):
    if reverse == True:
        if x < y:
            return x
        else:
            return 0
    else:
        if x < y:
            return 0
        else:
            return x

#############################
# Encoding/Decoding Utilities
#############################
def canon_smile(smile):
    """
    Transforms to canonic SMILES
    """
    return MolToSmiles(MolFromSmiles(smile))

def verified_and_below(smile, max_len):
    """
    Returns True if the SMILES string is valid and
    its length is less than max_len
    """
    return len(smile) < max_len and verify_sequence(smile)

def verify_sequence(smile):
    """
    Returns True if the SMILES string is valid and
    its length is less than max_len
    """
    mol = Chem.MolFromSmiles(smile)
    return smile != '' and mol is not None and mol.GetNumAtoms() > 1

def apply_to_valid(smile, fun, **kwargs):
    """
    Returns fun(smile, **kwargs) if smiles is a valid
    SMILES string, and 0.0 otherwise
    """
    mol = Chem.MolFromSmiles(smile)
    return fun(mol, **kwargs) if smile != '' and mol is not None and mol.GetNumAtoms() > 1 else 0.0

def filter_smiles(smiles):
    """
    Filters out valid SMILES string from a list
    """
    return [smile for smile in smiles if verify_sequence(smile)]

def build_vocab(smiles, pad_char='_', start_char='^'):
    """
    Builds the vocabulary dictionaries

    Arguments
    -----------
        - smiles: List of SMILES
        - pad_char: Char used for padding. '_' by default
        - start_char: First char of every generated string.
        '^' by default

    Returns
    -----------
        - char_dict: Dictionary which maps a given character
        to a number
        - ord_dict: Dictionary which maps a given number to a
        character
    """
    i = 1
    char_dict, ord_dict = {start_char: 0}, {0: start_char}
    for smile in smiles:
        for c in smile:
            if c not in char_dict:
                char_dict[c] = i
                ord_dict[i] = c
                i += 1
    char_dict[pad_char], ord_dict[i] = i, pad_char
    return char_dict, ord_dict

def pad(smile, n, pad_char='_'):
    """
    Adds the padding char (by default '_') to a string
    until it is of n length
    """
    if n < len(smile):
        return smile
    return smile + pad_char * (n - len(smile))

def unpad(smile, pad_char='_'): 
    """
    Removes the padding of a string
    """
    return smile.rstrip(pad_char)

def encode(smile, max_len, char_dict): 
    """
    Encodes a SMILES string using the previously built vocabulary
    """
    return [char_dict[c] for c in pad(smile, max_len)]

def decode(ords, ord_dict): 
    """
    Decodes a SMILES string using the previously built vocabulary
    """
    return unpad(''.join([ord_dict[o] for o in ords]))

def compute_results(model_samples, train_data, ord_dict, results={}, verbose=True):
    samples = [decode(s, ord_dict) for s in model_samples]
    results['mean_length'] = np.mean([len(sample) for sample in samples])
    results['n_samples'] = len(samples)
    results['uniq_samples'] = len(set(samples))
    verified_samples = []
    unverified_samples = []
    for sample in samples:
        if verify_sequence(sample):
            verified_samples.append(sample)
        else:
            unverified_samples.append(sample)
    results['good_samples'] = len(verified_samples)
    results['bad_samples'] = len(unverified_samples)
    # save smiles
    if 'Batch' in results.keys():
        smi_name = '{}_{}'.format(results['exp_name'], results['Batch'])
        save_smi(smi_name, samples)
        results['model_samples'] = smi_name
    if verbose:
        print_results(verified_samples, unverified_samples, results)
    return

def print_results(verified_samples, unverified_samples, results={}):
    print('Summary of the epoch')
    print('~~~~~~~~~~~~~~~~~~~~~~~~\n')
    print('{:15s} : {:6d}'.format("Total samples", results['n_samples']))
    percent = results['uniq_samples'] / float(results['n_samples']) * 100
    print('{:15s} : {:6d} ({:2.2f}%)'.format(
        'Unique', results['uniq_samples'], percent))
    percent = results['bad_samples'] / float(results['n_samples']) * 100
    print('{:15s} : {:6d} ({:2.2f}%)'.format('Unverified',
                                             results['bad_samples'], percent))
    percent = results['good_samples'] / float(results['n_samples']) * 100
    print('{:15s} : {:6d} ({:2.2f}%)'.format(
        'Verified', results['good_samples'], percent))

    print('\nSome good samples:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~\n')
    if len(verified_samples) > 10:
        for s in verified_samples[0:10]:
            print('' + s)
    else:
        print('No good samples were found :(...')

    print('\nSome bad samples:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~\n')
    if len(unverified_samples) > 10:
        for s in unverified_samples[0:10]:
            print('' + s)
    else:
        print('No bad samples were found :D!')

    return

##############
# Data Loaders
##############
class Gen_Dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def create_batches(self, samples):
        self.num_batch = int(len(samples) / self.batch_size)
        samples = samples[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(samples), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

class Dis_Dataloader():
    def __init__(self):
        self.vocab_size = 5000

    def load_data_and_labels(self, positive_examples, negative_examples):
        """
        Loads MR polarity data from files, splits the data into words and generates labels
        Returns split sentences and labels
        """
        # Split by words
        x_text = positive_examples + negative_examples

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)

        x_text = np.array(x_text)
        y = np.array(y)
        return [x_text, y]

    def load_train_data(self, positive_file, negative_file):
        """
        Returns input vectors, labels, vocabulary, and inverse vocabulary
        """
        # Load and preprocess data
        sentences, labels = self.load_data_and_labels(
            positive_file, negative_file)
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        x_shuffled = sentences[shuffle_indices]
        y_shuffled = labels[shuffle_indices]
        self.sequence_length = 20
        return [x_shuffled, y_shuffled]

    def load_test_data(self, positive_file, test_file):
        test_examples = []
        test_labels = []
        with open(test_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                test_examples.append(parse_line)
                test_labels.append([1, 0])

        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                test_examples.append(parse_line)
                test_labels.append([0, 1])

        test_examples = np.array(test_examples)
        test_labels = np.array(test_labels)
        shuffle_indices = np.random.permutation(np.arange(len(test_labels)))
        x_dev = test_examples[shuffle_indices]
        y_dev = test_labels[shuffle_indices]

        return [x_dev, y_dev]

    def batch_iter(self, data, batch_size, num_epochs):
        """
        Generates a batch iterator for a dataset
        """
        data = np.array(list(data))
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

###############
# Discriminator
###############
def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387)

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate
    """
    output = input_
    for idx in range(layer_size):
        with tf.variable_scope('output_lin_%d' % idx):
            output = f(core_rnn_cell._linear(output, size, 0))

        with tf.variable_scope('transform_lin_%d' % idx):
            transform_gate = tf.sigmoid(
                core_rnn_cell._linear(input_, size, 0) + bias)
            carry_gate = 1. - transform_gate

        output = transform_gate * output + carry_gate * input_

    return output

class Discriminator(object):
    """
    A CNN for text classification
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, 
                 filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(
                self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filter]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = sum(num_filters)
        self.h_pool = tf.concat(axis=3, values=pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add highway
        with tf.name_scope("highway"):
            self.h_highway = highway(
                self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(
                [num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.ypred_for_auc = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")

################
# Custom Metrics
################
def batch_validity(smiles, train_smiles=None):
    """
    Assigns 1.0 if the SMILES is correct, and 0.0 if not
    """
    vals = [1.0 if verify_sequence(s) else 0.0 for s in smiles]
    return vals

def batch_diversity(smiles, train_smiles):
    """
    Compares the Tanimoto distance of a given molecule
    with a random sample of the training smiles
    """
    rand_smiles = random.sample(train_smiles, 100)
    rand_mols = [MolFromSmiles(s) for s in rand_smiles]
    fps = [Chem.GetMorganFingerprintAsBitVect(
        m, 4, nBits=2048) for m in rand_mols]
    vals = [apply_to_valid(s, diversity, fps=fps) for s in smiles]
    return vals

def batch_variety(smiles, train_smiles=None):
    """
    Compares the Tanimoto distance of a given molecule
    with a random sample of the other generated smiles
    """
    filtered = filter_smiles(smiles)
    if filtered:
        mols = [Chem.MolFromSmiles(smile) for smile in
                np.random.choice(filtered, int(len(filtered) / 10))]
        setfps = [Chem.GetMorganFingerprintAsBitVect(
            mol, 4, nBits=2048) for mol in mols]
        vals = [apply_to_valid(s, variety, setfps=setfps) for s in smiles]
        return vals
    else:
        return np.zeros(len(smiles))

def batch_novelty(smiles, train_smiles):
    """
    Assigns 1.0 if the molecule is not in the training
    set, and 0.0 otherwise
    """
    vals = [novelty(smile, train_smiles) if verify_sequence(
        smile) else 0 for smile in smiles]
    return vals

def batch_hardnovelty(smiles, train_smiles):
    """
    Assigns 1.0 if the molecule's canonical SMILES is
    not in the training set, and 0.0 otherwise
    """
    vals = [hard_novelty(smile, train_smiles) if verify_sequence(
        smile) else 0 for smile in smiles]
    return vals

def batch_softnovelty(smiles, train_smiles):
    """
    Assigns 1.0 if the molecule is not in the training
    set, and 0.0 otherwise
    """
    vals = [soft_novelty(smile, train_smiles) if verify_sequence(
        smile) else 0 for smile in smiles]
    return vals

def batch_creativity(smiles, train_smiles):
    """
    Computes the Tanimoto distance of a smile
    to the training set, as a measure of how different
    these molecules are from the provided ones.
    """
    mols = [Chem.MolFromSmiles(smile) for smile in filter_smiles(train_smiles)]
    setfps = [Chem.GetMorganFingerprintAsBitVect(
        mol, 4, nBits=2048) for mol in mols]
    vals = [apply_to_valid(s, creativity, setfps=setfps) for s in smiles]
    return vals

def batch_symmetry(smiles, train_smiles=None):
    """
    Yields 1.0 if the generated molecule has 
    any element of symmetry, and 0.0 if the point group is C1
    """
    vals = [apply_to_valid(s, symmetry) for s in smiles]
    return vals

def batch_logP(smiles, train_smiles=None):
    """
    This metric computes the logarithm of the water-octanol partition
    coefficient, using RDkit's implementation of Wildman-Crippen method,
    and then remaps it to the 0.0-1.0 range.

    Wildman, S. A., & Crippen, G. M. (1999). 
    Prediction of physicochemical parameters by atomic contributions. 
    Journal of chemical information and computer sciences, 39(5), 868-873.
    """
    vals = [apply_to_valid(s, logP) for s in smiles]
    return vals

def batch_conciseness(smiles, train_smiles=None):
    """
    This metric penalizes SMILES strings that are too long, 
    assuming that the canonical representation is the shortest
    """
    vals = [conciseness(s) if verify_sequence(s) else 0 for s in smiles]
    return vals

def batch_lipinski(smiles, train_smiles):
    """
    This metric assigns 0.25 for every rule of Lipinski's
    rule-of-five that is obeyed
    """
    vals = [apply_to_valid(s, Lipinski) for s in smiles]
    return vals

def batch_SA(smiles, train_smiles=None, SA_model=None):
    """
    This metric checks whether a given molecule is easy to synthesize or not.
    It is based on (although not completely equivalent to) the work of Ertl
    and Schuffenhauer.

    Ertl, P., & Schuffenhauer, A. (2009).
    Estimation of synthetic accessibility score of drug-like molecules
    based on molecular complexity and fragment contributions.
    Journal of cheminformatics, 1(1), 8.
    """
    vals = [apply_to_valid(s, SA_score, SA_model=SA_model) for s in smiles]
    return vals

def batch_NPLikeliness(smiles, train_smiles=None, NP_model=None):
    """
    This metric computes the likelihood that a given molecule is
    a natural product
    """
    vals = [apply_to_valid(s, NP_score, NP_model=NP_model) for s in smiles]
    return vals

def batch_beauty(smiles, train_smiles=None):
    """
    Computes chemical beauty.

    Bickerton, G. R., Paolini, G. V., Besnard, J., Muresan, S., & Hopkins, A. L. (2012). 
    Quantifying the chemical beauty of drugs. 
    Nature chemistry, 4(2), 90-98.
    """
    vals = [apply_to_valid(s, chemical_beauty) for s in smiles]
    return vals

def batch_substructure_match_all(smiles, train_smiles=None, ALL_POS_PATTS=None):
    """
    Assigns 1.0 if all the specified substructures are present
    in the molecule
    """
    if ALL_POS_PATTS == None:
        raise ValueError('No substructures has been specified')

    vals = [apply_to_valid(s, substructure_match_all, ALL_POS_PATTS=ALL_POS_PATTS) for s in smiles]
    return vals

def batch_substructure_match_any(smiles, train_smiles=None, ANY_POS_PATTS=None):
    """
    Assigns 1.0 if any of the specified substructures are present
    in the molecule
    """
    if ANY_POS_PATTS == None:
        raise ValueError('No substructures has been specified')

    vals = [apply_to_valid(s, substructure_match_any, ANY_POS_PATTS=ANY_POS_PATTS) for s in smiles]
    return vals

def batch_substructure_absence(smiles, train_smiles=None, ALL_NEG_PATTS=None):
    """
    Assigns 0.0 if any of the substructures are present in the
    molecule, and 1.0 otherwise
    """
    if ALL_NEG_PATTS == None:
        raise ValueError('No substructures has been specified')

    vals = [apply_to_valid(s, substructure_match_any, ALL_NEG_PATTS=ALL_NEG_PATTS) for s in smiles]
    return vals

def batch_PCE(smiles, train_smiles=None, cnn=None):
    """
    Power conversion efficiency as computed by a neural network
    acting on Morgan fingerprints
    """
    if cnn == None:
        raise ValueError('The PCE metric was not properly loaded.')
    fsmiles = []
    zeroindex = []
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)
    vals = np.asarray(cnn.predict(fsmiles))
    for k in zeroindex:
        vals[k] = 0.0
    vals = np.squeeze(np.stack(vals, axis=1))
    return vals

def batch_bandgap(smiles, train_smiles=None, cnn=None):
    """
    HOMO-LUMO energy difference as computed by a neural network
    acting on Morgan fingerprints
    """
    if cnn == None:
        raise ValueError('The bandgap metric was not properly loaded.')
    fsmiles = []
    zeroindex = []
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)
    vals = np.asarray(cnn.predict(fsmiles))
    for k in zeroindex:
        vals[k] = 0.0
    vals = np.squeeze(np.stack(vals, axis=1))
    return vals

def batch_mp(smiles, train_smiles=None, cnn=None):
    """
    Melting point as computed by a neural network acting on 
    Morgan fingerprints
    """
    if cnn == None:
        raise ValueError('The melting point metric was not properly loaded.')
    fsmiles = []
    zeroindex = []
    for k, sm in enumerate(smiles):
        if verify_sequence(sm):
            fsmiles.append(sm)
        else:
            fsmiles.append('c1ccccc1')
            zeroindex.append(k)
    vals = np.asarray(cnn.predict(fsmiles))
    for k in zeroindex:
        vals[k] = 0.0
    vals = np.squeeze(np.stack(vals, axis=1))
    return vals

def batch_bp(smiles, train_smiles=None, gp=None):
   """
   Boiling point as computed by a gaussian process acting
   on Morgan fingerprints
   """
   if gp == None:
       raise ValueError('The boiling point metric was not properly loaded.')
   fsmiles = []
   zeroindex = []
   for k, sm in enumerate(smiles):
       if verify_sequence(sm):
           fsmiles.append(sm)
       else:
           fsmiles.append('c1ccccc1')
           zeroindex.append(k)
   vals = np.asarray(gp.predict(fsmiles))
   for k in zeroindex:
       vals[k] = 0.0
   vals = np.squeeze(np.stack(vals, axis=1))
   return vals

def batch_density(smiles, train_smiles=None, gp=None):
   """
   Density as computed by a gaussian process acting on
   Morgan fingerprints
   """
   if gp == None:
       raise ValueError('The density metric was not properly loaded.')
   fsmiles = []
   zeroindex = []
   for k, sm in enumerate(smiles):
       if verify_sequence(sm):
           fsmiles.append(sm)
       else:
           fsmiles.append('c1ccccc1')
           zeroindex.append(k)
   vals = np.asarray(gp.predict(fsmiles))
   for k in zeroindex:
       vals[k] = 0.0
   vals = np.squeeze(np.stack(vals, axis=1))
   return vals

def batch_mutagenicity(smiles, train_smiles=None, gp=None):
   """
   Mutagenicity as estimated by a gaussian process acting on
   Morgan fingerprints
   """
   if gp == None:
       raise ValueError('The mutagenicity was not properly loaded.')
   fsmiles = []
   zeroindex = []
   for k, sm in enumerate(smiles):
       if verify_sequence(sm):
           fsmiles.append(sm)
       else:
           fsmiles.append('c1ccccc1')
           zeroindex.append(k)
   vals = np.asarray(gp.predict(fsmiles))
   for k in zeroindex:
       vals[k] = 0.0
   vals = np.squeeze(np.stack(vals, axis=1))
   return vals

def batch_pvap(smiles, train_smiles=None, gp=None):
   """
   Vapour pressure as computed by a gaussian process acting on
   Morgan fingerprints
   """
   if gp == None:
       raise ValueError('The vapour pressure was not properly loaded.')
   fsmiles = []
   zeroindex = []
   for k, sm in enumerate(smiles):
       if verify_sequence(sm):
           fsmiles.append(sm)
       else:
           fsmiles.append('c1ccccc1')
           zeroindex.append(k)
   vals = np.asarray(gp.predict(fsmiles))
   for k in zeroindex:
       vals[k] = 0.0
   vals = np.squeeze(np.stack(vals, axis=1))
   return vals

def batch_solubility(smiles, train_smiles=None, gp=None):
   """
   Solubility in water as computed by a gaussian process acting on
   Morgan fingerprints
   """
   if gp == None:
       raise ValueError('The solubility was not properly loaded.')
   fsmiles = []
   zeroindex = []
   for k, sm in enumerate(smiles):
       if verify_sequence(sm):
           fsmiles.append(sm)
       else:
           fsmiles.append('c1ccccc1')
           zeroindex.append(k)
   vals = np.asarray(gp.predict(fsmiles))
   for k in zeroindex:
       vals[k] = 0.0
   vals = np.squeeze(np.stack(vals, axis=1))
   return vals

def batch_viscosity(smiles, train_smiles=None, gp=None):
   """
   Viscosity as computed by a gaussian process acting on
   Morgan fingerprints
   """
   if gp == None:
       raise ValueError('The viscosity was not properly loaded.')
   fsmiles = []
   zeroindex = []
   for k, sm in enumerate(smiles):
       if verify_sequence(sm):
           fsmiles.append(sm)
       else:
           fsmiles.append('c1ccccc1')
           zeroindex.append(k)
   vals = np.asarray(gp.predict(fsmiles))
   for k in zeroindex:
       vals[k] = 0.0
   vals = np.squeeze(np.stack(vals, axis=1))
   return vals

##################
# Loading Routines
##################
def load_NP(filename=None):
    """
    Loads the parameters required by the naturalness
    metric
    """
    if filename is None:
        filename = os.path.join(MOD_PATH, '../data/pkl/NP_score.pkl.gz')
    NP_model = pickle.load(gzip.open(filename))
    return ('NP_model', NP_model)

def load_SA(filename=None):
    """
    Loads the parameters required by the synthesizability
    metric
    """
    if filename is None:
        filename = os.path.join(MOD_PATH, '../data/pkl/SA_score.pkl.gz')
    model_data = pickle.load(gzip.open(filename))
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    SA_model = outDict
    return ('SA_model', SA_model)

def load_beauty(filename=None):
    """
    Loads the parameters required by the chemical
    beauty metric
    """
    if filename is None:
        filename = os.path.join(MOD_PATH, '../data/pkl/QED_score.pkl.gz')
    QED = pickle.load(gzip.open(filename))
    global AliphaticRings, Acceptors, StructuralAlerts, pads1, pads2
    AliphaticRings = QED[0]
    Acceptors = QED[1]
    StructuralAlerts = QED[2]
    pads1 = QED[3]
    pads2 = QED[4]
    return ('QED_model', QED_model)

def load_substructure_match_any():
    """
    Loads substructures for the 'MATCH ANY' metric
    """
    ANY_POS_PATTS = readSubstructuresFile('any_positive.smi', 'any_positive')
    return ('ANY_POS_PATTS', ANY_POS_PATTS)

def load_substructure_match_all():
    """
    Loads substructures for the 'MATCH ALL' metric
    """
    ALL_POS_PATTS = readSubstructuresFile('all_positive.smi', 'all_positive')
    return ('ALL_POS_PATTS', ALL_POS_PATTS)

def load_substructure_absence():
    """
    Loads substructures for the 'ABSENCE' metric
    """
    ALL_NEG_PATTS = readSubstructuresFile('all_negative.smi', 'all_negative')
    return ('ALL_NEG_PATTS', ALL_NEG_PATTS)

def load_PCE():
    """
    Loads the Keras NN model for Power Conversion
    Efficiency
    """
    cnn_pce = KerasNN('pce')
    cnn_pce.load('pce.h5')
    return ('cnn', cnn_pce)

def load_bandgap():
    """
    Loads the Keras NN model for HOMO-LUMO energy
    difference
    """
    cnn_bandgap = KerasNN('bandgap')
    cnn_bandgap.load('../data/nns/bandgap.h5')
    return ('cnn', cnn_bandgap)

def load_mp():
    """
    Loads the Keras NN model for melting point
    """
    cnn_mp = KerasNN('mp')
    cnn_mp.load('../data/nns/mp.h5')
    return ('cnn', cnn_mp)

def load_bp():
   """
   Loads the GPmol GP model for boiling point
   """
   gp_bp = GaussianProcess('bp')
   gp_bp.load('../data/gps/bp.json')
   return ('gp', gp_bp)

def load_density():
   """
   Loads the GPmol GP model for density
   """
   gp_density = GaussianProcess('density')
   gp_density.load('../data/gps/density.json')
   return ('gp', gp_density)

def load_mutagenicity():
   """
   Loads the GPmol GP model for mutagenicity
   """
   gp_mutagenicity = GaussianProcess('mutagenicity')
   gp_mutagenicity.load('../data/gps/mutagenicity.json')
   return ('gp', gp_mutagenicity)

def load_pvap():
   """
   Loads the GPmol GP model for vapour pressure
   """
   gp_pvap = GaussianProcess('pvap')
   gp_pvap.load('../data/gps/pvap.json')
   return ('gp', gp_pvap)

def load_solubility():
   """
   Loads the GPmol GP model for solubility
   """
   gp_solubility = GaussianProcess('solubility')
   gp_solubility.load('../data/gps/solubility.json')
   return ('gp', gp_solubility)

def load_viscosity():
   """
   Loads the GPmol GP model for viscosity
   """
   gp_viscosity = GaussianProcess('viscosity')
   gp_viscosity.load('../data/gps/viscosity.json')
   return ('gp', gp_viscosity)

##################
# Metric Functions
##################
def diversity(mol, fps):
    low_rand_dst = 0.9
    mean_div_dst = 0.945
    ref_fps = Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
    dist = DataStructs.BulkTanimotoSimilarity(
        ref_fps, fps, returnDistance=True)
    mean_dist = np.mean(np.array(dist))
    val = remap(mean_dist, low_rand_dst, mean_div_dst)
    val = np.clip(val, 0.0, 1.0)
    return val

def variety(mol, setfps):
    low_rand_dst = 0.9
    mean_div_dst = 0.945
    fp = Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
    dist = DataStructs.BulkTanimotoSimilarity(fp, setfps, returnDistance=True)
    mean_dist = np.mean(np.array(dist))
    return mean_dist

def novelty(smile, train_smiles):
    return 1.0 if smile not in train_smiles else 0.0

def hard_novelty(smile, train_smiles):
    return 1.0 if canon_smile(smile) not in train_smiles else 0.0

def soft_novelty(smile, train_smiles):
    return 1.0 if smile not in train_smiles else 0.3

def creativity(mol, setfps):
    return np.mean(DataStructs.BulkTanimotoSimilarity(Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048), setfps))

def symmetry(mol):
    try:
        ids, xyz = get3DCoords(mol)
        sch_symbol = getSymmetry(ids, xyz)
        return 1.0 if sch_symbol != 'C1' else 0.0
    except:
        return 0.0

def get3DCoords(mol):
    m = Chem.AddHs(mol)
    m.UpdatePropertyCache(strict=False)
    Chem.EmbedMolecule(m)
    Chem.MMFFOptimizeMolecule(m)
    molblock = Chem.MolToMolBlock(m)
    mblines = molblock.split('\n')[4:len(m.GetAtoms())]
    parsed = [entry.split() for entry in mblines]
    coords = [[coord[3], np.asarray([float(coord[0]), float(
        coord[1]), float(coord[2])])] for coord in parsed]
    ids = [coord[0] for coord in coords]
    xyz = [[coord[1][0], coord[1][1], coord[1][2]] for coord in coords]
    return ids, xyz

def getSymmetry(ids, xyz):
    mol = PointGroupAnalyzer(mg.Molecule(ids, xyz))
    return mol.sch_symbol

def logP(mol, train_smiles=None):
    val = Crippen.MolLogP(mol)
    return val

def SA_score(mol, SA_model):
    if SA_model is None:
        raise ValueError("Synthesizability metric was not properly loaded.")
    # fragment score
    fp = Chem.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += SA_model.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(
        mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    nSpiro = Chem.CalcNumSpiroAtoms(mol)
    nBridgeheads = Chem.CalcNumBridgeheadAtoms(mol)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - \
        spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
    val = remap(sascore, 5, 1.5)
    val = np.clip(val, 0.0, 1.0)
    return val

def conciseness(smile, train_smiles=None):
    canon = canon_smile(smile)
    diff_len = len(smile) - len(canon)
    val = np.clip(diff_len, 0.0, 20)
    val = 1 - 1.0 / 20.0 * val
    return val

def NP_score(mol, NP_model=None):
    fp = Chem.GetMorganFingerprint(mol, 2)
    bits = fp.GetNonzeroElements()

    # calculating the score
    val = 0.
    for bit in bits:
        val += NP_model.get(bit, 0)
    val /= float(mol.GetNumAtoms())

    # preventing score explosion for exotic molecules
    if val > 4:
        val = 4. + math.log10(val - 4. + 1.)
    if val < -4:
        val = -4. - math.log10(-4. - val + 1.)
    return val

def Lipinski(mol):
    druglikeness = 0.0
    druglikeness += 0.25 if logP(mol) <= 5 else 0.0
    druglikeness += 0.25 if rdkit.Chem.Descriptors.MolWt(mol) <= 500 else 0.0
    # Look for hydrogen bond aceptors
    acceptors = 0
    for atom in mol.GetAtoms():
        acceptors += 1 if atom.GetAtomicNum() == 8 else 0.0
        acceptors += 1 if atom.GetAtomicNum() == 7 else 0.0
    druglikeness += 0.25 if acceptors <= 10 else 0.0
    # Look for hydrogen bond donors
    donors = 0
    for bond in mol.GetBonds():
        a1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum()
        a2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum()
        donors += 1 if ((a1, a2) == (1, 8)) or ((a1, a2) == (8, 1)) else 0.0
        donors += 1 if ((a1, a2) == (1, 7)) or ((a1, a2) == (7, 1)) else 0.0
    druglikeness += 0.25 if donors <= 5 else 0.0
    return druglikeness

def substructure_match_all(mol, train_smiles=None):
    val = all([mol.HasSubstructMatch(patt) for patt in ALL_POS_PATTS])
    return int(val)

def substructure_match_any(mol, train_smiles=None):
    val = any([mol.HasSubstructMatch(patt) for patt in ANY_POS_PATTS])
    return int(val)

def substructure_absence(mol, train_smiles=None):
    val = all([not mol.HasSubstructMatch(patt) for patt in ANY_NEG_PATTS])
    return int(val)

def readSubstructuresFile(filename, label='positive'):
    if os.path.exists(filename):
        smiles = read_smi(filename)
        patterns = [Chem.MolFromSmarts(s) for s in smiles]
    else:
        patterns = None
    return patterns

def ads(x, a, b, c, d, e, f, dmax):
    return ((a + (b / (1 + exp(-1 * (x - c + d / 2) / e)) * (1 - 1 / (1 + exp(-1 * (x - c - d / 2) / f))))) / dmax)

def properties(mol):
    matches = []
    if (mol is None):
        raise WrongArgument("properties(mol)", "mol argument is \'None\'")
    x = [0] * 8
    # MW
    x[0] = Descriptors.MolWt(mol)
    # ALOGP
    x[1] = Descriptors.MolLogP(mol)
    for hba in Acceptors:                                                       # HBA
        if (mol.HasSubstructMatch(hba)):
            matches = mol.GetSubstructMatches(hba)
            x[2] += len(matches)
    x[3] = Descriptors.NumHDonors(
        mol)                                          # HBD
    # PSA
    x[4] = Descriptors.TPSA(mol)
    x[5] = Descriptors.NumRotatableBonds(
        mol)                                   # ROTB
    x[6] = Chem.GetSSSR(Chem.DeleteSubstructs(
        deepcopy(mol), AliphaticRings))   # AROM
    for alert in StructuralAlerts:                                              # ALERTS
        if (mol.HasSubstructMatch(alert)):
            x[7] += 1
    return x

def qed(w, p, gerebtzoff):
    d = [0.00] * 8
    if (gerebtzoff):
        for i in range(0, 8):
            d[i] = ads(p[i], pads1[i][0], pads1[i][1], pads1[i][2], pads1[
                       i][3], pads1[i][4], pads1[i][5], pads1[i][6])
    else:
        for i in range(0, 8):
            d[i] = ads(p[i], pads2[i][0], pads2[i][1], pads2[i][2], pads2[
                       i][3], pads2[i][4], pads2[i][5], pads2[i][6])
    t = 0.0
    for i in range(0, 8):
        t += w[i] * log(d[i])
    return (exp(t / sum(w)))

def weights_mean(mol, gerebtzoff=True):
    props = properties(mol)
    return qed([0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95], props, gerebtzoff)

def chemical_beauty(mol, gerebtzoff=True):
    return weights_mean(mol, gerebtzoff)

################
# Reward Loading
################
def metrics_loading():
    """
    Feeds the loading procedures to the main program
    """
    global MOD_PATH
    MOD_PATH = os.path.dirname(os.path.realpath(__file__))

    load = OrderedDict()
    # Cheminformatics
    load['validity'] = lambda *args: None
    load['diversity'] = lambda *args: None
    load['variety'] = lambda *args: None
    load['novelty'] = lambda *args: None
    load['hard_novelty'] = lambda *args: None
    load['soft_novelty'] = lambda *args: None
    load['creativity'] = lambda *args: None
    load['symmetry'] = lambda *args: None
    load['conciseness'] = lambda *args: None
    load['lipinski'] = lambda *args: None
    load['synthesizability'] = load_SA
    load['naturalness'] = load_NP
    load['chemical_beauty'] = load_beauty
    load['substructure_match_all'] = load_substructure_match_all
    load['substructure_match_any'] = load_substructure_match_any
    load['substructure_absence'] = load_substructure_absence
    load['mutagenicity'] = load_mutagenicity

    # Physical properties
    load['logP'] = lambda *args: None
    load['pce'] = load_PCE
    load['bandgap'] = load_bandgap
    load['mp'] = load_mp
    load['bp'] = load_bp
    load['density'] = load_density
    load['pvap'] = load_pvap
    load['solubility'] = load_solubility
    load['viscosity'] = load_viscosity

    return load

def get_metrics():
    """
    Feeds the metrics to the main program
    """
    metrics = OrderedDict()

    # Cheminformatics
    metrics['validity'] = batch_validity
    metrics['novelty'] = batch_novelty
    metrics['creativity'] = batch_creativity
    metrics['hard_novelty'] = batch_hardnovelty
    metrics['soft_novelty'] = batch_softnovelty
    metrics['diversity'] = batch_diversity
    metrics['variety'] = batch_variety
    metrics['symmetry'] = batch_symmetry
    metrics['conciseness'] = batch_conciseness
    metrics['lipinski'] = batch_lipinski
    metrics['synthesizability'] = batch_SA
    metrics['naturalness'] = batch_NPLikeliness
    metrics['chemical_beauty'] = batch_beauty
    metrics['substructure_match_all'] = batch_substructure_match_all
    metrics['substructure_match_any'] = batch_substructure_match_any
    metrics['substructure_absence'] = batch_substructure_absence
    metrics['mutagenicity'] = batch_mutagenicity

    # Physical properties
    metrics['logP'] = batch_logP
    metrics['pce'] = batch_PCE
    metrics['bandgap'] = batch_bandgap
    metrics['mp'] = batch_mp
    metrics['bp'] = batch_bp
    metrics['density'] = batch_density
    metrics['pvap'] = batch_pvap
    metrics['solubility'] = batch_solubility
    metrics['viscosity'] = batch_viscosity

    return metrics

##################
# Main ORGANIC
##################
class ORGANIC(object):
    """
    Main class: Where every interaction between
    the user and the backend is performed
    """
    def __init__(self, name, params={}, use_gpu=True, verbose=True):
        """
        Parameter initialization

        Arguments
        -----------
            - name: String which will be used to identify the
            model in any folders or files created
            - params: (Optional) Dictionary containing the parameters
            that the user whishes to specify
            - use_gpu: Boolean specifying whether a GPU should be
            used. True by default
            - verbose: Boolean specifying whether output must be
            produced in-line
        """
        self.verbose = verbose

        # Set minimum verbosity for RDKit, Keras, and TF backends
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.set_verbosity(logging.INFO)
        rdBase.DisableLog('rdApp.error')

        # Set configuration for GPU
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # Set model parameters
        self.PREFIX = name

        if 'PRETRAIN_GEN_EPOCHS' in params:
            self.PRETRAIN_GEN_EPOCHS = params['PRETRAIN_GEN_EPOCHS']
        else:
            self.PRETRAIN_GEN_EPOCHS = 240

        if 'PRETRAIN_DIS_EPOCHS' in params:
            self.PRETRAIN_DIS_EPOCHS = params['PRETRAIN_DIS_EPOCHS']
        else:
            self.PRETRAIN_DIS_EPOCHS = 50

        if 'GEN_ITERATIONS' in params:
            self.GEN_ITERATIONS = params['GEN_ITERATIONS']
        else:
            self.GEN_ITERATIONS = 2

        if 'GEN_BATCH_SIZE' in params:
            self.GEN_BATCH_SIZE = params['GEN_BATCH_SIZE']
        else:
            self.GEN_BATCH_SIZE = 64

        if 'SEED' in params:
            self.SEED = params['SEED']
        else:
            self.SEED = None
        random.seed(self.SEED)
        np.random.seed(self.SEED)

        if 'DIS_BATCH_SIZE' in params:
            self.DIS_BATCH_SIZE = params['DIS_BATCH_SIZE']
        else:
            self.DIS_BATCH_SIZE = 64

        if 'DIS_EPOCHS' in params:
            self.DIS_EPOCHS = params['DIS_EPOCHS']
        else:
            self.DIS_EPOCHS = 3

        if 'EPOCH_SAVES' in params:
            self.EPOCH_SAVES = params['EPOCH_SAVES']
        else:
            self.EPOCH_SAVES = 20

        if 'CHK_PATH' in params:
            self.CHK_PATH = params['CHK_PATH']
        else:
            self.CHK_PATH = os.path.join(
                os.getcwd(), 'checkpoints/{}'.format(self.PREFIX))

        if 'GEN_EMB_DIM' in params:
            self.GEN_EMB_DIM = params['GEN_EMB_DIM']
        else:
            self.GEN_EMB_DIM = 32

        if 'GEN_HIDDEN_DIM' in params:
            self.GEN_HIDDEN_DIM = params['GEN_HIDDEN_DIM']
        else:
            self.GEN_HIDDEN_DIM = 32

        if 'START_TOKEN' in params:
            self.START_TOKEN = params['START_TOKEN']
        else:
            self.START_TOKEN = 0

        if 'SAMPLE_NUM' in params:
            self.SAMPLE_NUM = params['SAMPLE_NUM']
        else:
            self.SAMPLE_NUM = 6400

        if 'BIG_SAMPLE_NUM' in params:
            self.BIG_SAMPLE_NUM = params['BIG_SAMPLE_NUM']
        else:
            self.BIG_SAMPLE_NUM = self.SAMPLE_NUM * 5

        if 'LAMBDA' in params:
            self.LAMBDA = params['LAMBDA']
        else:
            self.LAMBDA = 0.5

        # In case MAX_LENGTH is not specified by the user,
        # it will be determined later, in the training set
        # loading
        if 'MAX_LENGTH' in params:
            self.MAX_LENGTH = params['MAX_LENGTH']

        if 'DIS_EMB_DIM' in params:
            self.DIS_EMB_DIM = params['DIS_EMB_DIM']
        else:
            self.DIS_EMB_DIM = 64

        if 'DIS_FILTER_SIZES' in params:
            self.DIS_FILTER_SIZES = params['DIS_FILTER_SIZES']
        else:
            self.DIS_FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

        if 'DIS_NUM_FILTERS' in params:
            self.DIS_NUM_FILTERS = params['DIS_FILTER_SIZES']
        else:
            self.DIS_NUM_FILTERS = [100, 200, 200, 200, 200, 100,
                                    100, 100, 100, 100, 160, 160]

        if 'DIS_DROPOUT' in params:
            self.DIS_DROPOUT = params['DIS_DROPOUT']
        else:
            self.DIS_DROPOUT = 0.75
        if 'DIS_L2REG' in params:
            self.DIS_L2REG = params['DIS_L2REG']
        else:
            self.DIS_L2REG = 0.2

        self.AV_METRICS = get_metrics()
        self.LOADINGS = metrics_loading()

        self.PRETRAINED = False
        self.SESS_LOADED = False
        self.USERDEF_METRIC = False

    def load_training_set(self, file):
        """
        Specifies a training set for the model. It also finishes
        the model set up, as some of the internal parameters require
        knowledge of the vocabulary.

        Arguments
        -----------
            - fileL String pointing to a .smi or .csv file
        """

        # Load training set
        self.train_samples = load_train_data(file)

        # Process and create vocabulary
        self.char_dict, self.ord_dict = build_vocab(self.train_samples)
        self.NUM_EMB = len(self.char_dict)
        self.PAD_CHAR = self.ord_dict[self.NUM_EMB - 1]
        self.PAD_NUM = self.char_dict[self.PAD_CHAR]
        self.DATA_LENGTH = max(map(len, self.train_samples))

        # If MAX_LENGTH has not been specified by the user, it
        # will be set as 1.5 times the maximum length in the
        # trining set.
        if not hasattr(self, 'MAX_LENGTH'):
            self.MAX_LENGTH = int(len(max(self.train_samples, key=len)) * 1.5)

        # Encode samples
        to_use = [sample for sample in self.train_samples
                  if verified_and_below(sample, self.MAX_LENGTH)]
        self.positive_samples = [encode(sam, self.MAX_LENGTH, self.char_dict) for sam in to_use]
        self.POSITIVE_NUM = len(self.positive_samples)

        # Print information
        if self.verbose:

            print('\nPARAMETERS INFORMATION')
            print('============================\n')
            print('Model name               :   {}'.format(self.PREFIX))
            print('Training set size        :   {} points'.format(
                len(self.train_samples)))
            print('Max data length          :   {}'.format(self.MAX_LENGTH))
            print('Avg length to use is     :   {}'.format(
                np.mean([len(s) for s in to_use])))
            print('Num valid data points is :   {}'.format(
                self.POSITIVE_NUM))
            print('Size of alphabet is      :   {}'.format(self.NUM_EMB))
            print('')

            params = ['PRETRAIN_GEN_EPOCHS', 'PRETRAIN_DIS_EPOCHS',
                      'GEN_ITERATIONS', 'GEN_BATCH_SIZE', 'SEED',
                      'DIS_BATCH_SIZE', 'DIS_EPOCHS', 'EPOCH_SAVES',
                      'CHK_PATH', 'GEN_EMB_DIM', 'GEN_HIDDEN_DIM',
                      'START_TOKEN', 'SAMPLE_NUM', 'BIG_SAMPLE_NUM',
                      'LAMBDA', 'MAX_LENGTH', 'DIS_EMB_DIM',
                      'DIS_FILTER_SIZES', 'DIS_NUM_FILTERS',
                      'DIS_DROPOUT', 'DIS_L2REG']

            for param in params:
                string = param + ' ' * (25 - len(param))
                value = getattr(self, param)
                print('{}:   {}'.format(string, value))

        # Set model
        self.gen_loader = Gen_Dataloader(self.GEN_BATCH_SIZE)
        self.dis_loader = Dis_Dataloader()
        self.mle_loader = Gen_Dataloader(self.GEN_BATCH_SIZE)
        self.generator = Generator(self.NUM_EMB, self.GEN_BATCH_SIZE,
                                   self.GEN_EMB_DIM, self.GEN_HIDDEN_DIM,
                                   self.MAX_LENGTH, self.START_TOKEN)

        with tf.variable_scope('discriminator'):
            self.discriminator = Discriminator(
                sequence_length=self.MAX_LENGTH,
                num_classes=2,
                vocab_size=self.NUM_EMB,
                embedding_size=self.DIS_EMB_DIM,
                filter_sizes=self.DIS_FILTER_SIZES,
                num_filters=self.DIS_NUM_FILTERS,
                l2_reg_lambda=self.DIS_L2REG)
        self.dis_params = [param for param in tf.trainable_variables()
                           if 'discriminator' in param.name]
        self.dis_global_step = tf.Variable(
            0, name="global_step", trainable=False)
        self.dis_optimizer = tf.train.AdamOptimizer(1e-4)
        self.dis_grads_and_vars = self.dis_optimizer.compute_gradients(
            self.discriminator.loss, self.dis_params, aggregation_method=2)
        self.dis_train_op = self.dis_optimizer.apply_gradients(
            self.dis_grads_and_vars, global_step=self.dis_global_step)

        self.sess = tf.Session(config=self.config)
        self.folder = 'checkpoints/{}'.format(self.PREFIX)

    def define_metric(self, name, metric, load_metric=lambda *args: None,
                      pre_batch=False, pre_metric=lambda *args: None):
        """
        Sets up a new metric and generates a .pkl file in
        the data/ directory

        Arguments
        -----------
            - name: String used to identify the metric
            - metric: Function taking as argument a SMILES
            string and returning a float value
            - load_metric: (Optional) Preprocessing needed
            at the beginning of the code
            - pre_batch: (Optional) Boolean specifying whether
            there is any preprocessing when the metric is applied
            to a batch of smiles. False by default.
            - pre_metric: (Optional) Preprocessing operations
            for the metric. Will be ignored if pre_batch is False.

        Notes
        -----------
            - For combinations of already existing metrics, check
            the define_metric_as_combination method.
            - For metrics based in neural networks or gaussian
            processes, please check our more specific functions
            define_nn_metric and define_gp_metric.
            - Check the mol_methods module for useful processing
            options, and the custom_metrics module for examples
            of how metrics are defined in ORGANIC.
        """

        if pre_batch:
            def batch_metric(smiles, train_smiles=None):
                psmiles = pre_metric()
                vals = [apply_to_valid(s, metric) for s in psmiles]
                return vals
        else:
            def batch_metric(smiles, train_smiles=None):
                vals = [apply_to_valid(s, metric) for s in smiles]
                return vals

        self.AV_METRICS[name] = batch_metric
        self.LOADINGS[name] = load_metric

        if self.verbose:
            print('Defined metric {}'.format(name))

        metric = [batch_metric, load_metric]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(metric, f)

    def define_metric_as_combination(self, name, metrics, ponderations):
        """
        Sets up a metric made from a combination of
        previously existing metrics. Also generates a
        metric .pkl file in the data/ directory.

        Arguments
        -----------
            - nameL String used to identify the metric
            - metrics: List containing the name identifiers
            of every metric in the list
            - ponderations: List of ponderation coefficients
            for every metric in the previous list
        """

        funs = [self.AV_METRICS[metric] for metric in metrics]
        funs_load = [self.LOADINGS[metric] for metric in metrics]

        def metric(smiles, train_smiles=None, **kwargs):
            vals = np.zeros(len(smiles))
            for fun, c in zip(funs, ponderations):
                vals += c * np.asarray(fun(smiles))
            return vals

        def load_metric():
            return [fun() for fun in funs_load if fun() is not None]

        self.AV_METRICS[name] = metric
        self.LOADINGS[name] = load_metric

        if self.verbose:
            print('Defined metric {}'.format(name))

        nmetric = [metric, load_metric]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(nmetric, f)

    def define_metric_as_remap(self, name, metric, remapping):
        """
        Sets up a metric made from a remapping of a
        previously existing metric. Also generates a .pkl
        metric file in the data/ directory

        Arguments
        -----------
            - name: String used to identify the metric
            - metric: String identifying the previous metric
            - remapping: Remap function

        Notes
        -----------
            Use of the mathematical remappings provided in the
            mol_methods module is highly recommended
        """

        pmetric = self.AV_METRICS[metric]

        def nmetric(smiles, train_smiles=None, **kwargs):
            vals = pmetric(smiles, train_smiles, **kwargs)
            return remapping(vals)

        self.AV_METRICS[name] = nmetric
        self.LOADINGS[name] = self.LOADINGS[metric]

        if self.verbose:
            print('Defined metric {}'.format(name))

        metric = [nmetric, self.LOADINGS[metric]]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(metric, f)

    def train_nn_as_metric(self, name, train_x, train_y, nepochs=1000):
        """
        Sets up a metric with a neural network trained on
        a dataset

        Arguments.
        -----------
            - name: String used to identify the metric
            - train_x: List of SMILES identificators
            - train_y: List of property values
            - nepochs: Number of epochs for training

        Notes
        -----------
            A name.h5 file is generated in the data/nns directory,
            and this metric can be loaded in the future using the
            load_prev_user_metric() method through the name.pkl
            file generated in the data/ dir.

                load_prev_user_metric('name.pkl')
        """
        cnn = KerasNN(name)
        cnn.train(train_x, train_y, 500, nepochs, earlystopping=True, min_delta=0.001)
        K.clear_session()

        def batch_NN(smiles, train_smiles=None, nn=None):
            """
            User-trained neural network
            """
            if nn == None:
                raise ValueError('The user-trained NN metric was not properly loaded.')
            fsmiles = []
            zeroindex = []
            for k, sm in enumerate(smiles):
                if verify_sequence(sm):
                    fsmiles.append(sm)
                else:
                    fsmiles.append('c1ccccc1')
                    zeroindex.append(k)
            vals = np.asarray(nn.predict(fsmiles))
            for k in zeroindex:
                vals[k] = 0.0
            vals = np.squeeze(np.stack(vals, axis=1))
            return vals

        def load_NN():
            """
            Loads the Keras NN model for a user-trained metric
            """
            nn = KerasNN(name)
            nn.load('../data/nns/{}.h5'.format(name))
            return ('nn', nn)

        self.AV_METRICS[name] = batch_NN
        self.LOADINGS[name] = load_NN

        if self.verbose:
            print('Defined metric {}'.format(name))

        metric = [batch_NN, load_NN]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(metric, f)

    def load_prev_user_metric(self, name, file=None):
        """
        Loads a metric that the user has previously designed

        Arguments.
        -----------
            - name: String used to identify the metric
            - file: String pointing to the .pkl file. Will use
            data/name.pkl by default.
        """
        if file is None:
            file = '../data/{}.pkl'.format(name)
        pkl = open(file, 'rb')
        data = pickle.load(pkl)
        self.AV_METRICS[name] = data[0]
        self.LOADINGS[name] = data[1]
        if self.verbose:
            print('Loaded metric {}'.format(name))

    def set_training_program(self, metrics=None, steps=None):
        """
        Sets a program of metrics and epochs
        for training the model and generating molecules

        Arguments
        -----------
            - metrics: List of metrics. Each element represents
            the metric used with a particular set of epochs. Its
            length must coincide with the steps list.
            - steps: List of epoch sets. Each element represents
            the number of epochs for which a given metric will
            be used. Its length must coincide with the steps list.

        Note
        -----------
            The program will crash if both lists have different
            lengths

        Example
        -----------
            The following examples trains the model for, sequentially,
            20 epochs of PCE, 100 epochs of bandgap and another 20
            epochs of PCE.
                model = ORGANIC('model')
                model.load_training_set('sample.smi')
                model.set_training_program(['pce', 'bandgap', 'pce'],
                                           [20, 100, 20])
        """
        # Raise error if the lengths do not match
        if len(metrics) != len(steps):
            return ValueError('Unmatching lengths in training program.')

        # Set important parameters
        self.TOTAL_BATCH = np.sum(np.asarray(steps))
        self.METRICS = metrics

        # Build the 'educative program'
        self.EDUCATION = {}
        i = 0
        for j, stage in enumerate(steps):
            for _ in range(stage):
                self.EDUCATION[i] = metrics[j]
                i += 1

    def load_metrics(self):
        """
        Loads the metrics
        """

        # Get the list of used metrics
        met = list(set(self.METRICS))

        # Execute the metrics loading
        self.kwargs = {}
        for m in met:
            load_fun = self.LOADINGS[m]
            args = load_fun()
            if args is not None:
                if isinstance(args, tuple):
                    self.kwargs[m] = {args[0]: args[1]}
                elif isinstance(args, list):
                    fun_args = {}
                    for arg in args:
                        fun_args[arg[0]] = arg[1]
                    self.kwargs[m] = fun_args
            else:
                self.kwargs[m] = None

    def load_prev_pretraining(self, ckpt=None):
        """
        Loads a previous pretraining

        Arguments
        -----------
            - ckpt: String pointing to the ckpt file. By default,
            'checkpoints/name_pretrain/pretrain_ckpt' is assumed.

        Note
        -----------
            The models are stored by the Tensorflow API backend. This
            will generate various files, like in the following ls:

                checkpoint
                pretrain_ckpt.data-00000-of-00001
                pretrain_ckpt.index
                pretrain_ckpt.meta

            In this case, ckpt = 'pretrain_ckpt'.

        Note 2
        -----------
            Due to its structure, ORGANIC is very dependent on its
            hyperparameters (for example, MAX_LENGTH defines the
            embedding). Most of the errors with this function are
            related to parameter mismatching.
        """
        # Generate TF saver
        saver = tf.train.Saver()

        # Set default checkpoint
        if ckpt is None:
            ckpt_dir = 'checkpoints/{}_pretrain'.format(self.PREFIX)
            if not os.path.exists(ckpt_dir):
                print('No pretraining data was found')
                return
            ckpt = os.path.join(ckpt_dir, 'pretrain_ckpt')

        # Load from checkpoint
        if os.path.isfile(ckpt + '.meta'):
            saver.restore(self.sess, ckpt)
            print('Pretrain loaded from previous checkpoint {}'.format(ckpt))
            self.PRETRAINED = True
        else:
            print('\t* No pre-training data found as {:s}.'.format(ckpt))

    def load_prev_training(self, ckpt=None):
        """
        Loads a previous trained model

        Arguments
        -----------
            - ckpt: String pointing to the ckpt file. By default,
            'checkpoints/name/pretrain_ckpt' is assumed.

        Note 1
        -----------
            The models are stored by the Tensorflow API backend. This
            will generate various files. An example ls:

                checkpoint
                validity_model_0.ckpt.data-00000-of-00001
                validity_model_0.ckpt.index
                validity_model_0.ckpt.meta
                validity_model_100.ckpt.data-00000-of-00001
                validity_model_100.ckpt.index
                validity_model_100.ckpt.meta
                validity_model_120.ckpt.data-00000-of-00001
                validity_model_120.ckpt.index
                validity_model_120.ckpt.meta
                validity_model_140.ckpt.data-00000-of-00001
                validity_model_140.ckpt.index
                validity_model_140.ckpt.meta

                    ...

                validity_model_final.ckpt.data-00000-of-00001
                validity_model_final.ckpt.index
                validity_model_final.ckpt.meta

            Possible ckpt values are 'validity_model_0', 'validity_model_140'
            or 'validity_model_final'.

        Note 2
        -----------
            Due to its structure, ORGANIC is very dependent on its
            hyperparameters (for example, MAX_LENGTH defines the
            embedding). Most of the errors with this function are
            related to parameter mismatching.

        """
        # If there is no Rollout, add it
        if not hasattr(self, 'rollout'):
            self.rollout = Rollout(self.generator, 0.8, self.PAD_NUM)

        # Generate TF Saver
        saver = tf.train.Saver()

        # Set default checkpoint
        if ckpt is None:
            ckpt_dir = 'checkpoints/{}'.format(self.PREFIX)
            if not os.path.exists(ckpt_dir):
                print('No pretraining data was found')
                return
            ckpt = os.path.join(ckpt_dir, 'pretrain_ckpt')

        if os.path.isfile(ckpt + '.meta'):
            saver.restore(self.sess, ckpt)
            print('Training loaded from previous checkpoint {}'.format(ckpt))
            self.SESS_LOADED = True
        else:
            print('\t* No training checkpoint found as {:s}.'.format(ckpt))

    def pretrain(self):
        """
        Pretrains generator and discriminator
        """
        self.gen_loader.create_batches(self.positive_samples)
        # results = OrderedDict({'exp_name': self.PREFIX})

        if self.verbose:
            print('\nPRETRAINING')
            print('============================\n')
            print('GENERATOR PRETRAINING')

        for epoch in tqdm(range(self.PRETRAIN_GEN_EPOCHS)):

            supervised_g_losses = []
            self.gen_loader.reset_pointer()

            for it in range(self.gen_loader.num_batch):
                batch = self.gen_loader.next_batch()
                _, g_loss, g_pred = self.generator.pretrain_step(self.sess,
                                                                 batch)
                supervised_g_losses.append(g_loss)
            loss = np.mean(supervised_g_losses)

            if epoch % 10 == 0:

                print('\t train_loss {}'.format(loss))

        samples = self.generate_samples(self.SAMPLE_NUM)
        self.mle_loader.create_batches(samples)

        if self.LAMBDA != 0:

            if self.verbose:
                print('\nDISCRIMINATOR PRETRAINING')

            for i in tqdm(range(self.PRETRAIN_DIS_EPOCHS)):

                negative_samples = self.generate_samples(self.POSITIVE_NUM)
                dis_x_train, dis_y_train = self.dis_loader.load_train_data(
                    self.positive_samples, negative_samples)
                dis_batches = self.dis_loader.batch_iter(
                    zip(dis_x_train, dis_y_train), self.DIS_BATCH_SIZE,
                    self.PRETRAIN_DIS_EPOCHS)

                for batch in dis_batches:
                    x_batch, y_batch = zip(*batch)
                    feed = {
                        self.discriminator.input_x: x_batch,
                        self.discriminator.input_y: y_batch,
                        self.discriminator.dropout_keep_prob: self.DIS_DROPOUT
                    }
                    _, step, loss, accuracy = self.sess.run(
                        [self.dis_train_op, self.dis_global_step,
                         self.discriminator.loss, self.discriminator.accuracy],
                        feed)

        self.PRETRAINED = True

    def generate_samples(self, num):
        """
        Generates molecules

        Arguments
        -----------
            - num: Integer representing the number of molecules
        """
        generated_samples = []

        for _ in range(int(num / self.GEN_BATCH_SIZE)):
            generated_samples.extend(self.generator.generate(self.sess))

        return generated_samples

    def train(self, ckpt_dir='checkpoints/'):
        """
        Trains the model 
        If necessary, also includes pretraining
        """
        if not self.PRETRAINED and not self.SESS_LOADED:

            self.sess.run(tf.global_variables_initializer())
            self.pretrain()

            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            ckpt_file = os.path.join(ckpt_dir,
                                     '{}_pretrain_ckpt'.format(self.PREFIX))
            saver = tf.train.Saver()
            path = saver.save(self.sess, ckpt_file)
            if self.verbose:
                print('Pretrain saved at {}'.format(path))

        if not hasattr(self, 'rollout'):
            self.rollout = Rollout(self.generator, 0.8, self.PAD_NUM)

        if self.verbose:
            print('\nSTARTING TRAINING')
            print('============================\n')

        results_rows = []
        for nbatch in tqdm(range(self.TOTAL_BATCH)):

            results = OrderedDict({'exp_name': self.PREFIX})

            metric = self.EDUCATION[nbatch]

            if metric in self.AV_METRICS.keys():
                reward_func = self.AV_METRICS[metric]
            else:
                raise ValueError('Metric {} not found!'.format(metric))

            if self.kwargs[metric] is not None:

                def batch_reward(samples):
                    decoded = [decode(sample, self.ord_dict)
                               for sample in samples]
                    pct_unique = len(list(set(decoded))) / float(len(decoded))
                    rewards = reward_func(decoded, self.train_samples,
                                          **self.kwargs[metric])
                    weights = np.array([pct_unique /
                                        float(decoded.count(sample))
                                        for sample in decoded])

                    return rewards * weights

            else:

                def batch_reward(samples):
                    decoded = [decode(sample, self.ord_dict)
                               for sample in samples]
                    pct_unique = len(list(set(decoded))) / float(len(decoded))
                    rewards = reward_func(decoded, self.train_samples)
                    weights = np.array([pct_unique /
                                        float(decoded.count(sample))
                                        for sample in decoded])

                    return rewards * weights

            if nbatch % 10 == 0:
                gen_samples = self.generate_samples(self.BIG_SAMPLE_NUM)
            else:
                gen_samples = self.generate_samples(self.SAMPLE_NUM)
            self.gen_loader.create_batches(gen_samples)
            results['Batch'] = nbatch
            print('Batch n. {}'.format(nbatch))
            print('============================\n')

            # results
            compute_results(
                gen_samples, self.train_samples, self.ord_dict, results)

            for it in range(self.GEN_ITERATIONS):
                samples = self.generator.generate(self.sess)
                rewards = self.rollout.get_reward(
                    self.sess, samples, 16, self.discriminator,
                    batch_reward, self.LAMBDA)
                nll = self.generator.generator_step(
                    self.sess, samples, rewards)

                print('Rewards')
                print('~~~~~~~~~~~~~~~~~~~~~~~~\n')
                np.set_printoptions(precision=3, suppress=True)
                mean_r, std_r = np.mean(rewards), np.std(rewards)
                min_r, max_r = np.min(rewards), np.max(rewards)
                print('Mean:                {:.3f}'.format(mean_r))
                print('               +/-   {:.3f}'.format(std_r))
                print('Min:                 {:.3f}'.format(min_r))
                print('Max:                 {:.3f}'.format(max_r))
                np.set_printoptions(precision=8, suppress=False)
                results['neg-loglike'] = nll
            self.rollout.update_params()

            # generate for discriminator
            if self.LAMBDA != 0:
                print('\nDISCRIMINATOR TRAINING')
                print('============================\n')
                for i in range(self.DIS_EPOCHS):
                    print('Discriminator epoch {}...'.format(i+1))

                    negative_samples = self.generate_samples(self.POSITIVE_NUM)
                    dis_x_train, dis_y_train = self.dis_loader.load_train_data(
                        self.positive_samples, negative_samples)
                    dis_batches = self.dis_loader.batch_iter(
                        zip(dis_x_train, dis_y_train),
                        self.DIS_BATCH_SIZE, self.DIS_EPOCHS
                    )

                    for batch in dis_batches:
                        x_batch, y_batch = zip(*batch)
                        feed = {
                            self.discriminator.input_x: x_batch,
                            self.discriminator.input_y: y_batch,
                            self.discriminator.dropout_keep_prob:
                                self.DIS_DROPOUT
                        }
                        _, step, d_loss, accuracy = self.sess.run(
                            [self.dis_train_op, self.dis_global_step,
                             self.discriminator.loss,
                             self.discriminator.accuracy],
                            feed)

                    results['D_loss_{}'.format(i)] = d_loss
                    results['Accuracy_{}'.format(i)] = accuracy
                print('\nDiscriminator trained.')
            results_rows.append(results)
            if nbatch % self.EPOCH_SAVES == 0 or \
               nbatch == self.TOTAL_BATCH - 1:

                if results_rows is not None:
                    df = pd.DataFrame(results_rows)
                    df.to_csv('{}_results.csv'.format(self.folder),
                              index=False)
                if nbatch is None:
                    label = 'final'
                else:
                    label = str(nbatch)

                # save models
                model_saver = tf.train.Saver()
                ckpt_dir = os.path.join(self.CHK_PATH, self.folder)
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                ckpt_file = os.path.join(
                    ckpt_dir, '{}_{}.ckpt'.format(self.PREFIX, label))
                path = model_saver.save(self.sess, ckpt_file)
                print('\nModel saved at {}'.format(path))

        print('\n######### FINISHED #########')

