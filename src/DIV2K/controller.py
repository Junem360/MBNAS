from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time

import numpy as np
import tensorflow as tf

from src.utils import get_train_ops
from src.ops_general import stack_lstm

from tensorflow.python.training import moving_averages


class ControllerNetwork():
    def __init__(self,
                 search_whole_channels=False,
                 num_branches=3,
                 num_cells=6,
                 num_layers=3,
                 lstm_size=32,
                 lstm_num_layers=2,
                 lstm_keep_prob=1.0,
                 tanh_constant=None,
                 op_tanh_reduce=1.0,
                 temperature=None,
                 lr_init=1e-3,
                 lr_dec_start=0,
                 lr_dec_every=100,
                 lr_dec_rate=0.9,
                 l2_reg=0,
                 entropy_weight=None,
                 clip_mode=None,
                 grad_bound=None,
                 use_critic=False,
                 bl_dec=0.999,
                 optim_algo="adam",
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 name="controller",
                 use_cb_reward=False,
                 cb_rate=0,
                 **kwargs):

        print("-" * 80)
        print("Building ConvController")
        self.use_cb_reward = use_cb_reward
        self.cb_rate=cb_rate
        self.num_cells = num_cells
        self.num_layers = num_layers
        self.num_branches = num_branches
        self.seq_length = self.num_cells*2*self.num_branches
        self.num_controllers = self.seq_length

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce
        self.temperature = temperature
        self.lr_init = lr_init
        self.lr_dec_start = lr_dec_start
        self.lr_dec_every = lr_dec_every
        self.lr_dec_rate = lr_dec_rate
        self.l2_reg = l2_reg
        self.entropy_weight = entropy_weight
        self.clip_mode = clip_mode
        self.grad_bound = grad_bound
        self.bl_dec = bl_dec

        self.optim_algo = optim_algo
        self.sync_replicas = sync_replicas
        self.num_aggregate = num_aggregate
        self.num_replicas = num_replicas
        self.name = name

        self._create_params()
        arc_seq, entropy, log_prob, _, _ = self._build_sampler(use_bias=False)
        self.sample_arc = arc_seq
        self.sample_entropy = entropy
        self.sample_log_prob = log_prob


    def _create_params(self):
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(self.name, initializer=initializer):
            with tf.variable_scope("lstm"):
                self.w_lstm = []
                for layer_id in range(self.lstm_num_layers):
                    with tf.variable_scope("layer_{}".format(layer_id)):
                        w = tf.get_variable("w", [2 * self.lstm_size, 4 * self.lstm_size])
                        self.w_lstm.append(w)

            self.g_emb = tf.get_variable("g_emb", [1, self.lstm_size])

            with tf.variable_scope("emb"):
                self.op_emb = tf.get_variable("op_emb", [self.num_branches, self.lstm_size])

            with tf.variable_scope("softmax"):
                self.op_weights = tf.get_variable("idx_w", [self.num_controllers,self.lstm_size,self.num_branches])
                self.op_biases = tf.get_variable("idx_b", [self.num_controllers, self.num_branches], initializer=tf.constant_initializer(0))

                # self.op_weight = tf.get_variable("op_w", [self.lstm_size, 2])
                # b_init = np.array([0] * 2, dtype=np.float32)
                # self.b_soft = tf.get_variable("op_b", [1, 2], initializer=tf.constant_initializer(b_init))

                b_soft_no_learn = np.array(
                    [0.0, 0.0], dtype=np.float32)
                b_soft_no_learn = np.reshape(b_soft_no_learn, [1, 2])
                self.b_soft_no_learn = tf.constant(b_soft_no_learn, dtype=tf.float32)

    def _build_sampler(self, prev_c=None, prev_h=None, use_bias=False):
        """Build the sampler ops and the log_prob ops."""

        print("-" * 80)
        print("Build controller sampler")

        # anchors = tf.TensorArray(
        #     tf.float32, size=self.num_cells + 2, clear_after_read=False)
        # anchors_w_1 = tf.TensorArray(
        #     tf.float32, size=self.num_cells + 2, clear_after_read=False)

        arc_seq = tf.TensorArray(tf.int32, size=self.seq_length)
        # arc_seq = tf.TensorArray(tf.int32, size=50)
        if prev_c is None:
            assert prev_h is None, "prev_c and prev_h must both be None"
            prev_c = [tf.zeros([1, self.lstm_size], tf.float32)
                      for _ in range(self.lstm_num_layers)]
            prev_h = [tf.zeros([1, self.lstm_size], tf.float32)
                      for _ in range(self.lstm_num_layers)]
        inputs = self.g_emb

        # for layer_id in range(2):
        #     next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
        #     prev_c, prev_h = next_c, next_h
        #     anchors = anchors.write(layer_id, tf.zeros_like(next_h[-1]))
        #     anchors_w_1 = anchors_w_1.write(
        #         layer_id, tf.matmul(next_h[-1], self.w_attn_1))

        def _condition(controller_id, *args):
            return tf.less(controller_id, self.num_controllers)

        def _body(controller_id, inputs, prev_c, prev_h, arc_seq,
                  entropy, log_prob):
            # indices = tf.range(0, layer_id, dtype=tf.int32)
            start_id = controller_id
            prev_layers = []
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            # controller_id = tf.Print(controller_id,[controller_id],message="controller id : ")
            op_weight = tf.gather(self.op_weights, controller_id)
            op_weight = tf.reshape(op_weight,[self.lstm_size, self.num_branches])
            op_bias = tf.gather(self.op_biases, controller_id)
            op_bias = tf.reshape(op_bias,[1,self.num_branches])
            logits = tf.matmul(next_h[-1], op_weight) + op_bias
            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                op_tanh = self.tanh_constant / self.op_tanh_reduce
                logits = op_tanh * tf.tanh(logits)
            if use_bias:
                logits += self.b_soft_no_learn
            # logits = tf.Print(logits,[layer_id, logits],message="logits of controller in layer ", summarize=-1)
            # op_id = tf.reshape(logits,[-1])
            # op_id = tf.arg_max(op_id, 0)
            op_id = tf.multinomial(logits, 1)
            op_id = tf.to_int32(op_id)
            op_id = tf.reshape(op_id, [1])
            arc_seq = arc_seq.write(start_id, op_id)
            curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=op_id)
            log_prob += curr_log_prob
            curr_ent = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.nn.softmax(logits)))
            entropy += curr_ent
            # print(multiply_const)
            # cell_type = tf.Print(cell_type, [cell_type],message="cell type of controller before reduce sum")
            # cell_type = tf.Print(cell_type, [cell_type],message="cell type of controller")
            inputs = tf.nn.embedding_lookup(self.op_emb, op_id)
            inputs = tf.reshape(inputs, [1, self.lstm_size])
            # anchors = anchors.write(layer_id, next_h[-1])
            # anchors_w_1 = anchors_w_1.write(layer_id, tf.matmul(next_h[-1], self.w_attn_1))
            # inputs = self.g_emb

            return (controller_id + 1, inputs, next_c, next_h,
                    arc_seq, entropy, log_prob)

        loop_vars = [
            tf.constant(0, dtype=tf.int32, name="controller_id"),
            inputs,
            prev_c,
            prev_h,
            arc_seq,
            tf.constant([0.0], dtype=tf.float32, name="entropy"),
            tf.constant([0.0], dtype=tf.float32, name="log_prob"),
        ]

        loop_outputs = tf.while_loop(_condition, _body, loop_vars,
                                     parallel_iterations=1)
        controller_id = loop_outputs[0]
        # controller_id = tf.Print(controller_id, [controller_id], message="controller id : ")
        inputs = loop_outputs[1]
        prev_c = loop_outputs[2]
        prev_h = loop_outputs[3]
        arc_seq = loop_outputs[4]
        entropy = loop_outputs[5]
        log_prob = loop_outputs[6]

        arc_seq = arc_seq.stack()
        arc_seq = tf.reshape(arc_seq, [3,-1])
        entropy = tf.reduce_sum(entropy)
        log_prob = tf.reduce_sum(log_prob)
        last_c = prev_c
        last_h = prev_h

        return arc_seq, entropy, log_prob, last_c, last_h

    def build_trainer(self, child_model):
        child_model.build_valid_rl()
        self.valid_PSNR = tf.placeholder(dtype=tf.float32)
        self.complexity_based_reward = tf.placeholder(dtype=tf.float32)
        self.reward = self.valid_PSNR
        # self.reward = tf.Print(self.reward, [self.reward], message="reward of this batch : ")
        if self.use_cb_reward:
            self.reward = self.reward - self.cb_rate*self.complexity_based_reward
        if self.entropy_weight is not None:
            self.reward += self.entropy_weight * self.sample_entropy

        self.sample_log_prob = tf.reduce_sum(self.sample_log_prob)
        # self.sample_log_prob = tf.Print(self.sample_log_prob,[self.sample_log_prob], message="sample_log_prob of this batch : ")
        self.baseline = tf.placeholder(dtype=tf.float32)
        # baseline_update = tf.assign_sub(
        #     self.baseline, (1 - self.bl_dec) * (self.baseline - self.reward))
        #
        # with tf.control_dependencies([baseline_update]):
        #     self.reward = tf.identity(self.reward)

        self.loss = self.sample_log_prob * (self.reward - self.baseline)
        # self.loss = tf.Print(self.loss,[self.loss], message="loss of this batch : ")
        self.train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="train_step")

        tf_variables = [var for var in tf.trainable_variables() if var.name.startswith(self.name)]
        print("-" * 80)
        for var in tf_variables:
            print(var)

        self.train_op, self.lr, self.grad_norm, self.optimizer, self.grads = get_train_ops(
            self.loss,
            tf_variables,
            self.train_step,
            clip_mode=self.clip_mode,
            grad_bound=self.grad_bound,
            l2_reg=self.l2_reg,
            lr_init=self.lr_init,
            lr_dec_start=self.lr_dec_start,
            lr_dec_every=self.lr_dec_every,
            lr_dec_rate=self.lr_dec_rate,
            optim_algo=self.optim_algo,
            sync_replicas=self.sync_replicas,
            num_aggregate=self.num_aggregate,
            num_replicas=self.num_replicas)

        self.skip_rate = tf.constant(0.0, dtype=tf.float32)
