from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

from src.DIV2K.models import Model
from src.DIV2K.image_ops import Channel_attention
from src.DIV2K.image_ops import Pixel_shuffler
from src.DIV2K.image_ops import Texture_attention

from src.utils import count_model_params
from src.utils import get_train_ops
from src.ops_general import create_weight
from src.ops_general import create_bias


class ChildNetwork(Model):
    def __init__(self,
                 images,
                 labels,
                 meta_data,
                 output_dir = "./output",
                 use_aux_heads=False,
                 use_model=None,
                 fine_tune=False,
                 feature_fusion=False,
                 channel_attn = False,
                 cutout_size=None,
                 fixed_arcs=None,
                 upsample_size=2,
                 num_layers=2,
                 num_cells=5,
                 out_filters=24,
                 sfe_filters=64,
                 # keep_prob=1.0,
                 # drop_path_keep_prob=None,
                 batch_size=32,
                 eval_batch_size=100,
                 test_batch_size=1,
                 clip_mode=None,
                 grad_bound=None,
                 l2_reg=1e-4,
                 lr_init=1e-4,
                 lr_dec_start=0,
                 lr_warmup_val = None,
                 lr_warmup_steps = 5,
                 lr_dec_every=10000,
                 lr_dec_rate=0.1,
                 lr_dec_min=1e-5,
                 lr_cosine=False,
                 lr_max=None,
                 lr_min=None,
                 lr_T_0=None,
                 lr_T_mul=None,
                 num_epochs=None,
                 it_per_epoch=1000,
                 optim_algo=None,
                 sync_replicas=False,
                 num_branches=3,
                 num_aggregate=None,
                 num_replicas=None,
                 data_format="NHWC",
                 name="child",
                 **kwargs
                 ):
        """
        """

        super(self.__class__, self).__init__(
            images,
            labels,
            meta_data,
            output_dir=output_dir,
            cutout_size=cutout_size,
            use_model=use_model,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            test_batch_size=test_batch_size,
            clip_mode=clip_mode,
            grad_bound=grad_bound,
            l2_reg=l2_reg,
            lr_init=lr_init,
            it_per_epoch=it_per_epoch,
            lr_dec_start=lr_dec_start,
            lr_warmup_val = lr_warmup_val,
            lr_warmup_steps = lr_warmup_steps,
            lr_dec_every=lr_dec_every,
            lr_dec_rate=lr_dec_rate,
            lr_dec_min=lr_dec_min,
            # keep_prob=keep_prob,
            optim_algo=optim_algo,
            sync_replicas=sync_replicas,
            num_aggregate=num_aggregate,
            num_replicas=num_replicas,
            data_format=data_format,
            name=name)

        if self.data_format == "NHWC":
            self.actual_data_format = "channels_last"
        elif self.data_format == "NCHW":
            self.actual_data_format = "channels_first"
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

        self.use_model = use_model
        self.fine_tune = fine_tune
        self.use_aux_heads = use_aux_heads
        self.num_epochs = num_epochs
        self.num_train_steps = self.num_epochs * self.num_train_batches
        self.channel_attn = channel_attn
        # self.drop_path_keep_prob = drop_path_keep_prob
        self.lr_cosine = lr_cosine
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_T_0 = lr_T_0
        self.lr_T_mul = lr_T_mul
        self.upsample_size = upsample_size
        self.out_filters = out_filters
        self.sfe_filters = sfe_filters
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.num_branches = num_branches
        # self.now_arc = tf.TensorArray(tf.int32,size=0,dynamic_size=True)

        self.fixed_arcs = fixed_arcs
        if fixed_arcs is not None:
            self.exist_fixed_arc = True
        else:
            self.exist_fixed_arc = False
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="global_step")
        self.texture_map = tf.placeholder(dtype=tf.float32)
        # if self.drop_path_keep_prob is not None:
        #     assert num_epochs is not None, "Need num_epochs to drop_path"

        if self.use_aux_heads:
            self.aux_head_indices = self.num_layers // 2
        
    def _get_C(self, x):
        """
        Args:
          x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        if self.data_format == "NHWC":
            return x.get_shape()[3].value
        elif self.data_format == "NCHW":
            return x.get_shape()[1].value
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _get_HW(self, x):
        """
        Args:
          x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        return x.get_shape()[2].value

    def _get_strides(self, stride):
        """
        Args:
          x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        if self.data_format == "NHWC":
            return [1, stride, stride, 1]
        elif self.data_format == "NCHW":
            return [1, 1, stride, stride]
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _model(self, images, is_training, reuse=False):
        """Compute the predictions given the images."""
        # self.now_arc = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        with tf.variable_scope(self.name, reuse=reuse):
            # the first two inputs
            with tf.variable_scope("stem_conv"):
                # w = create_weight("w", [3, 3, 3, self.out_filters * 3])
                w = create_weight("w_grl", [3, 3, 3, self.sfe_filters])
                b = create_bias("b_grl", [self.sfe_filters])
                x_sfe = tf.nn.conv2d(
                    images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                print("Layer x_sfe: {}".format(x_sfe))
            if self.data_format == "NHWC":
                split_axis = 3
            elif self.data_format == "NCHW":
                split_axis = 1
            else:
                raise ValueError("Unknown data_format '{0}'".format(self.data_format))

            x = [x_sfe, x_sfe, x_sfe]

            # building layers in the micro space
            out_filters = self.out_filters
            block_outputs = []
            block_outputs_Low = []
            block_outputs_Mid = []
            block_outputs_High = []
            block_outputs.append(block_outputs_Low)
            block_outputs.append(block_outputs_Mid)
            block_outputs.append(block_outputs_High)

            for layer_id in range(self.num_layers):
                with tf.variable_scope("layer_{0}".format(layer_id)):
                    # print("fixed_arcs = {}".format(self.fixed_arcs))
                    if self.exist_fixed_arc:
                        x = self._fixed_block(x, self.fixed_arcs, out_filters, is_training)
                    else:
                        x = self._dnas_block(x, self.fixed_arcs, out_filters)
                    # x = tf.Print(x, [1], message="x : ")
                    block_outputs[0].append(x[0])
                    block_outputs[1].append(x[1])
                    block_outputs[2].append(x[2])
                    print("Layer_Low {0:>2d}: {1}".format(layer_id, x[0]))
                    print("Layer_Mid {0:>2d}: {1}".format(layer_id, x[1]))
                    print("Layer_High {0:>2d}: {1}".format(layer_id, x[2]))

            with tf.variable_scope("gff_coef"):
                alpha_gff_Low = tf.get_variable("alpha_Low", [1], initializer=tf.constant_initializer(1/3),
                                                trainable=True)
                alpha_gff_Mid = tf.get_variable("alpha_Mid", [1], initializer=tf.constant_initializer(1 / 3),
                                                trainable=True)
                alpha_gff_High = tf.get_variable("alpha_High", [1], initializer=tf.constant_initializer(1 / 3),
                                                trainable=True)

            with tf.variable_scope("Low_branch"):
                out_Low = tf.concat(block_outputs[0], axis=3)
                w_gff_Low = create_weight("w_gff", [3, 3, self.out_filters*self.num_layers, self.out_filters])
                b_gff_Low = create_bias("b_gff", [self.out_filters])
                out_Low = tf.nn.conv2d(
                    out_Low, w_gff_Low, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b_gff_Low

            with tf.variable_scope("Mid_branch"):
                out_Mid = tf.concat(block_outputs[1], axis=3)
                w_gff_Mid = create_weight("w_gff", [3, 3, self.out_filters*self.num_layers, self.out_filters])
                b_gff_Mid = create_bias("b_gff", [self.out_filters])
                out_Mid = tf.nn.conv2d(
                    out_Mid, w_gff_Mid, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b_gff_Mid

            with tf.variable_scope("High_branch"):
                out_High = tf.concat(block_outputs[2], axis=3)
                w_gff_High = create_weight("w_gff", [3, 3, self.out_filters*self.num_layers, self.out_filters])
                b_gff_High = create_bias("b_gff", [self.out_filters])
                out_High = tf.nn.conv2d(
                    out_High, w_gff_High, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b_gff_High

            res_out = tf.add_n([out_Low*alpha_gff_Low, out_Mid*alpha_gff_Mid, out_High*alpha_gff_High])
            with tf.variable_scope("res_ps"):
                res_out = Pixel_shuffler(res_out, self.out_filters, 3, self.data_format, self.upsample_size)
            with tf.variable_scope("lr_ps"):
                lr_images = Pixel_shuffler(images, 3, 3, self.data_format, self.upsample_size)
            out = lr_images + res_out
            print("Layer final_x: {}".format(out))

        return out

    def _fixed_block(self, x, arcs, out_filters, is_training):
        x_ssc = x
        cell_outputs = []
        cell_outputs_Low = []
        cell_outputs_Mid = []
        cell_outputs_High = []
        cell_outputs.append(cell_outputs_Low)
        cell_outputs.append(cell_outputs_Mid)
        cell_outputs.append(cell_outputs_High)

        for cell_id in range(self.num_cells):
            start_id= 2*(cell_id)
            end_id = 2*(cell_id+1)
            with tf.variable_scope("cell_{0}".format(cell_id)):
                x_connections = []
                for i in range(3):
                    x_connection = arcs[i][start_id:end_id]
                    x_connections.append(x_connection)
                # print("x_connections = {}".format(x_connections))
                x = self._fixed_cell(x, cell_id, x_connections)
                cell_outputs[0].append(x[0])
                cell_outputs[1].append(x[1])
                cell_outputs[2].append(x[2])

        # print("layers in fixed block: {}".format(layers))
        with tf.variable_scope("lff_coef"):
            alpha_lff_skip_Low = tf.get_variable("alpha_skip_Low", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_lff_skip_Mid = tf.get_variable("alpha_skip_Mid", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_lff_skip_High = tf.get_variable("alpha_skip_High", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_lff_res_Low = tf.get_variable("alpha_res_Low", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_lff_res_Mid = tf.get_variable("alpha_res_Mid", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_lff_res_High = tf.get_variable("alpha_res_High", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
        # print("Layer x_l_concat: {}".format(x_l_concat))
        with tf.variable_scope("Low_branch"):
            out_Low = tf.concat(cell_outputs[0], axis=3)
            w_lff_Low = create_weight("w_lff", [3, 3, self.out_filters*self.num_cells, self.out_filters])
            b_lff_Low = create_bias("b_lff", [self.out_filters])
            out_Low = tf.nn.conv2d(
                out_Low, w_lff_Low, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b_lff_Low
            out_Low = tf.add_n([out_Low*alpha_lff_res_Low, x_ssc[0]*alpha_lff_skip_Low])

        with tf.variable_scope("Mid_branch"):
            out_Mid = tf.concat(cell_outputs[1], axis=3)
            w_lff_Mid = create_weight("w_lff", [3, 3, self.out_filters*self.num_cells, self.out_filters])
            b_lff_Mid = create_bias("b_lff", [self.out_filters])
            out_Mid = tf.nn.conv2d(
                out_Mid, w_lff_Mid, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b_lff_Mid
            out_Mid = tf.add_n([out_Mid*alpha_lff_res_Mid, x_ssc[1]*alpha_lff_skip_Mid])

        with tf.variable_scope("High_branch"):
            out_High = tf.concat(cell_outputs[2], axis=3)
            w_lff_High = create_weight("w_lff", [3, 3, self.out_filters*self.num_cells, self.out_filters])
            b_lff_High = create_bias("b_lff", [self.out_filters])
            out_High = tf.nn.conv2d(
                out_High, w_lff_High, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b_lff_High
            out_High = tf.add_n([out_High*alpha_lff_res_High, x_ssc[2]*alpha_lff_skip_High])



        out = []
        out.append(out_Low)
        out.append(out_Mid)
        out.append(out_High)

        return out

    def _fixed_cell(self, x, cell_id, connection):
        """Performs an enas operation specified by op_id."""
        # op_connections = np.reshape(connection, [op_num, cell_id + 1])
        x_res = x
        w = {}
        b = {}
        for i in range(2):
            for j in range(self.num_branches):
                if i == 0:
                    w["w_op{}_{}".format(j, i)] = create_weight("w_op{}_{}".format(j, i),
                                                                [3, 3, self.out_filters, self.out_filters*4])
                    b["b_op{}_{}".format(j, i)] = create_bias("b_op{}_{}".format(j, i), [self.out_filters*4])
                elif i == 1:
                    w["w_op{}_{}".format(j, i)] = create_weight("w_op{}_{}".format(j, i),
                                                                [3, 3, self.out_filters*4, self.out_filters])
                    b["b_op{}_{}".format(j, i)] = create_bias("b_op{}_{}".format(j, i), [self.out_filters])

        with tf.variable_scope("unit_coef"):
            alpha_skip_Low = tf.get_variable("alpha_skip_Low", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_skip_Mid = tf.get_variable("alpha_skip_Mid", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_skip_High = tf.get_variable("alpha_skip_High", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_res_Low = tf.get_variable("alpha_res_Low", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_res_Mid = tf.get_variable("alpha_res_Mid", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_res_High = tf.get_variable("alpha_res_High", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)

        with tf.variable_scope("Low_branch"):
            x_Low = x[0]
            tmp_w = w["w_op{}_{}".format(connection[0][0],0)]
            tmp_b = b["b_op{}_{}".format(connection[0][0],0)]
            if connection[0][0] == 0:
                x_Low = tf.nn.conv2d(x_Low, tmp_w, [1, 1, 1, 1], "SAME") + tmp_b
                x_Low = tf.nn.relu(x_Low)
            elif connection[0][0] == 1:
                x_Low = tf.nn.atrous_conv2d(x_Low, filters=tmp_w, rate=2, padding="SAME") + tmp_b
                x_Low = tf.nn.relu(x_Low)
            elif connection[0][0] == 2:
                x_Low = tf.nn.atrous_conv2d(x_Low, filters=tmp_w, rate=3, padding="SAME") + tmp_b
                x_Low = tf.nn.relu(x_Low)

            tmp_w = w["w_op{}_{}".format(connection[0][1],1)]
            tmp_b = b["b_op{}_{}".format(connection[0][1],1)]
            if connection[0][1] == 0:
                x_Low = tf.nn.conv2d(x_Low, tmp_w, [1, 1, 1, 1], "SAME") + tmp_b
            elif connection[0][1] == 1:
                x_Low = tf.nn.atrous_conv2d(x_Low, filters=tmp_w, rate=2, padding="SAME") + tmp_b
            elif connection[0][1] == 2:
                x_Low = tf.nn.atrous_conv2d(x_Low, filters=tmp_w, rate=3, padding="SAME") + tmp_b

            out_Low = Channel_attention(x_Low, 16, self.out_filters, self.data_format)
            out_Low = tf.add_n([out_Low*alpha_res_Low, x_res[0]*alpha_skip_Low])

        with tf.variable_scope("Mid_branch"):
            x_Mid = x[1]
            tmp_w = w["w_op{}_{}".format(connection[1][0], 0)]
            tmp_b = b["b_op{}_{}".format(connection[1][0], 0)]
            if connection[1][0] == 0:
                x_Mid = tf.nn.conv2d(x_Mid, tmp_w, [1, 1, 1, 1], "SAME") + tmp_b
                x_Mid = tf.nn.relu(x_Mid)
            elif connection[1][0] == 1:
                x_Mid = tf.nn.atrous_conv2d(x_Mid, filters=tmp_w, rate=2, padding="SAME") + tmp_b
                x_Mid = tf.nn.relu(x_Mid)
            elif connection[1][0] == 2:
                x_Mid = tf.nn.atrous_conv2d(x_Mid, filters=tmp_w, rate=3, padding="SAME") + tmp_b
                x_Mid = tf.nn.relu(x_Mid)

            tmp_w = w["w_op{}_{}".format(connection[1][1], 1)]
            tmp_b = b["b_op{}_{}".format(connection[1][1], 1)]
            if connection[1][1] == 0:
                x_Mid = tf.nn.conv2d(x_Mid, tmp_w, [1, 1, 1, 1], "SAME") + tmp_b
            elif connection[1][1] == 1:
                x_Mid = tf.nn.atrous_conv2d(x_Mid, filters=tmp_w, rate=2, padding="SAME") + tmp_b
            elif connection[1][1] == 2:
                x_Mid = tf.nn.atrous_conv2d(x_Mid, filters=tmp_w, rate=3, padding="SAME") + tmp_b

            out_Mid = Channel_attention(x_Mid, 16, self.out_filters, self.data_format)
            out_Mid = tf.add_n([out_Mid * alpha_res_Mid, x_res[1] * alpha_skip_Mid])
        with tf.variable_scope("High_branch"):
            x_High = x[2]
            tmp_w = w["w_op{}_{}".format(connection[2][0], 0)]
            tmp_b = b["b_op{}_{}".format(connection[2][0], 0)]
            if connection[2][0] == 0:
                x_High = tf.nn.conv2d(x_High, tmp_w, [1, 1, 1, 1], "SAME") + tmp_b
                x_High = tf.nn.relu(x_High)
            elif connection[2][0] == 1:
                x_High = tf.nn.atrous_conv2d(x_High, filters=tmp_w, rate=2, padding="SAME") + tmp_b
                x_High = tf.nn.relu(x_High)
            elif connection[2][0] == 2:
                x_High = tf.nn.atrous_conv2d(x_High, filters=tmp_w, rate=3, padding="SAME") + tmp_b
                x_High = tf.nn.relu(x_High)

            tmp_w = w["w_op{}_{}".format(connection[2][1], 1)]
            tmp_b = b["b_op{}_{}".format(connection[2][1], 1)]
            if connection[2][1] == 0:
                x_High = tf.nn.conv2d(x_High, tmp_w, [1, 1, 1, 1], "SAME") + tmp_b
            elif connection[2][1] == 1:
                x_High = tf.nn.atrous_conv2d(x_High, filters=tmp_w, rate=2, padding="SAME") + tmp_b
            elif connection[2][1] == 2:
                x_High = tf.nn.atrous_conv2d(x_High, filters=tmp_w, rate=3, padding="SAME") + tmp_b

            out_High = Channel_attention(x_High, 16, self.out_filters, self.data_format)
            out_High = tf.add_n([out_High * alpha_res_High, x_res[2] * alpha_skip_High])
        out = []
        out.append(out_Low)
        out.append(out_Mid)
        out.append(out_High)

        return out

    def _dnas_block(self, x, arcs, out_filters):
        x_ssc = x
        cell_outputs = []
        cell_outputs_Low = []
        cell_outputs_Mid = []
        cell_outputs_High = []
        cell_outputs.append(cell_outputs_Low)
        cell_outputs.append(cell_outputs_Mid)
        cell_outputs.append(cell_outputs_High)

        for cell_id in range(self.num_cells):
            start_id= 2*(cell_id)
            end_id = 2*(cell_id+1)
            with tf.variable_scope("cell_{0}".format(cell_id)):
                x_connections = []
                for i in range(3):
                    x_connection = arcs[i][start_id:end_id]
                    x_connections.append(x_connection)
                # print("x_connections = {}".format(x_connections))
                x = self._dnas_cell(x, cell_id, x_connections)
                cell_outputs[0].append(x[0])
                cell_outputs[1].append(x[1])
                cell_outputs[2].append(x[2])

        # print("layers in fixed block: {}".format(layers))
        with tf.variable_scope("lff_coef"):
            alpha_lff_skip_Low = tf.get_variable("alpha_skip_Low", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_lff_skip_Mid = tf.get_variable("alpha_skip_Mid", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_lff_skip_High = tf.get_variable("alpha_skip_High", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_lff_res_Low = tf.get_variable("alpha_res_Low", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_lff_res_Mid = tf.get_variable("alpha_res_Mid", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_lff_res_High = tf.get_variable("alpha_res_High", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
        # print("Layer x_l_concat: {}".format(x_l_concat))
        with tf.variable_scope("Low_branch"):
            out_Low = tf.concat(cell_outputs[0], axis=3)
            w_lff_Low = create_weight("w_lff", [3, 3, self.out_filters*self.num_cells, self.out_filters])
            b_lff_Low = create_bias("b_lff", [self.out_filters])
            out_Low = tf.nn.conv2d(
                out_Low, w_lff_Low, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b_lff_Low
            out_Low = tf.add_n([out_Low*alpha_lff_res_Low, x_ssc[0]*alpha_lff_skip_Low])

        with tf.variable_scope("Mid_branch"):
            out_Mid = tf.concat(cell_outputs[1], axis=3)
            w_lff_Mid = create_weight("w_lff", [3, 3, self.out_filters*self.num_cells, self.out_filters])
            b_lff_Mid = create_bias("b_lff", [self.out_filters])
            out_Mid = tf.nn.conv2d(
                out_Mid, w_lff_Mid, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b_lff_Mid
            out_Mid = tf.add_n([out_Mid*alpha_lff_res_Mid, x_ssc[1]*alpha_lff_skip_Mid])

        with tf.variable_scope("High_branch"):
            out_High = tf.concat(cell_outputs[2], axis=3)
            w_lff_High = create_weight("w_lff", [3, 3, self.out_filters*self.num_cells, self.out_filters])
            b_lff_High = create_bias("b_lff", [self.out_filters])
            out_High = tf.nn.conv2d(
                out_High, w_lff_High, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b_lff_High
            out_High = tf.add_n([out_High*alpha_lff_res_High, x_ssc[2]*alpha_lff_skip_High])



        out = []
        out.append(out_Low)
        out.append(out_Mid)
        out.append(out_High)

        return out


    def _dnas_cell(self, x, cell_id, connection):
        """Performs an enas operation specified by op_id."""
        x_res = x
        w = {}
        b = {}
        for i in range(2):
            for j in range(self.num_branches):
                if i == 0:
                    w["w_op{}_{}".format(j, i)] = create_weight("w_op{}_{}".format(j, i),
                                                                [3, 3, self.out_filters, self.out_filters*4])
                    b["b_op{}_{}".format(j, i)] = create_bias("b_op{}_{}".format(j, i), [self.out_filters*4])
                elif i == 1:
                    w["w_op{}_{}".format(j, i)] = create_weight("w_op{}_{}".format(j, i),
                                                                [3, 3, self.out_filters*4, self.out_filters])
                    b["b_op{}_{}".format(j, i)] = create_bias("b_op{}_{}".format(j, i), [self.out_filters])

        with tf.variable_scope("unit_coef"):
            alpha_skip_Low = tf.get_variable("alpha_skip_Low", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_skip_Mid = tf.get_variable("alpha_skip_Mid", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_skip_High = tf.get_variable("alpha_skip_High", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_res_Low = tf.get_variable("alpha_res_Low", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_res_Mid = tf.get_variable("alpha_res_Mid", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)
            alpha_res_High = tf.get_variable("alpha_res_High", [1], initializer=tf.constant_initializer(1),
                                            trainable=True)

        with tf.variable_scope("Low_branch"):
            x_Low = x[0]
            x_Low_op1 = self._dnas_conv(x_Low, w, b, 0)
            x_Low_op2 = self._dnas_dilated_conv(x_Low, w, b, 0, 2)
            x_Low_op3 = self._dnas_dilated_conv(x_Low, w, b, 0, 3)
            x_Low = tf.case({tf.equal(connection[0][0], 0): lambda: x_Low_op1,
                      tf.equal(connection[0][0], 1): lambda: x_Low_op2,
                      tf.equal(connection[0][0], 2): lambda: x_Low_op3})
            x_Low = tf.nn.relu(x_Low)
            x_Low_op1 = self._dnas_conv(x_Low, w, b, 1)
            x_Low_op2 = self._dnas_dilated_conv(x_Low, w, b, 1, 2)
            x_Low_op3 = self._dnas_dilated_conv(x_Low, w, b, 1, 3)
            x_Low = tf.case({tf.equal(connection[0][1], 0): lambda: x_Low_op1,
                      tf.equal(connection[0][1], 1): lambda: x_Low_op2,
                      tf.equal(connection[0][1], 2): lambda: x_Low_op3})

            out_Low = Channel_attention(x_Low, 16, self.out_filters, self.data_format)
            out_Low = tf.add_n([out_Low * alpha_res_Low, x_res[0] * alpha_skip_Low])

        with tf.variable_scope("Mid_branch"):
            x_Mid = x[1]
            x_Mid_op1 = self._dnas_conv(x_Mid, w, b, 0)
            x_Mid_op2 = self._dnas_dilated_conv(x_Mid, w, b, 0, 2)
            x_Mid_op3 = self._dnas_dilated_conv(x_Mid, w, b, 0, 3)
            x_Mid = tf.case({tf.equal(connection[1][0], 0): lambda: x_Mid_op1,
                      tf.equal(connection[1][0], 1): lambda: x_Mid_op2,
                      tf.equal(connection[1][0], 2): lambda: x_Mid_op3})
            x_Mid = tf.nn.relu(x_Mid)
            x_Mid_op1 = self._dnas_conv(x_Mid, w, b, 1)
            x_Mid_op2 = self._dnas_dilated_conv(x_Mid, w, b, 1, 2)
            x_Mid_op3 = self._dnas_dilated_conv(x_Mid, w, b, 1, 3)
            x_Mid = tf.case({tf.equal(connection[1][1], 0): lambda: x_Mid_op1,
                      tf.equal(connection[1][1], 1): lambda: x_Mid_op2,
                      tf.equal(connection[1][1], 2): lambda: x_Mid_op3})

            out_Mid = Channel_attention(x_Mid, 16, self.out_filters, self.data_format)
            out_Mid = tf.add_n([out_Mid * alpha_res_Mid, x_res[1] * alpha_skip_Mid])

        with tf.variable_scope("High_branch"):
            x_High = x[2]
            x_High_op1 = self._dnas_conv(x_High, w, b, 0)
            x_High_op2 = self._dnas_dilated_conv(x_High, w, b, 0, 2)
            x_High_op3 = self._dnas_dilated_conv(x_High, w, b, 0, 3)
            x_High = tf.case({tf.equal(connection[2][0], 0): lambda: x_High_op1,
                      tf.equal(connection[2][0], 1): lambda: x_High_op2,
                      tf.equal(connection[2][0], 2): lambda: x_High_op3})
            x_High = tf.nn.relu(x_High)
            x_High_op1 = self._dnas_conv(x_High, w, b, 1)
            x_High_op2 = self._dnas_dilated_conv(x_High, w, b, 1, 2)
            x_High_op3 = self._dnas_dilated_conv(x_High, w, b, 1, 3)
            x_High = tf.case({tf.equal(connection[2][1], 0): lambda: x_High_op1,
                      tf.equal(connection[2][1], 1): lambda: x_High_op2,
                      tf.equal(connection[2][1], 2): lambda: x_High_op3})

            out_High = Channel_attention(x_High, 16, self.out_filters, self.data_format)
            out_High = tf.add_n([out_High * alpha_res_High, x_res[2] * alpha_skip_High])

        out = []
        out.append(out_Low)
        out.append(out_Mid)
        out.append(out_High)

        return out

    def _dnas_conv(self, x, w, b, pos):
        tmp_w = w["w_op{}_{}".format(0, pos)]
        tmp_b = b["b_op{}_{}".format(0, pos)]

        out = tf.nn.conv2d(x, tmp_w, [1, 1, 1, 1], "SAME") + tmp_b

        return out

    def _dnas_dilated_conv(self, x, w, b, pos, rate):
        tmp_w = w["w_op{}_{}".format(rate-1, pos)]
        tmp_b = b["b_op{}_{}".format(rate-1, pos)]

        out = tf.nn.atrous_conv2d(x, filters=tmp_w, rate=rate, padding="SAME") + tmp_b

        return out

    # override
    def _build_train(self):
        print("-" * 80)
        print("Build train graph")

        self.train_preds = self._model(self.x_train, True)
        self.loss = tf.losses.absolute_difference(labels=self.y_train, predictions=self.train_preds)
        # self.loss = tf.Print(self.loss, [self.loss], message="self.loss changed! : ")
        # self.loss = tf.losses.mean_squared_error(labels=self.y_train, predictions=self.train_preds)

        # self.train_PSNR = tf.image.psnr(self.y_valid, preds, 1)
        # self.train_PSNR = tf.reduce_sum(self.valid_PSNR)
        # self.aux_loss = tf.losses.mean_squared_error(labels=self.y_train, predictions=self.aux_preds)
        train_loss = self.loss

        tf_variables = [
            var for var in tf.trainable_variables() if var.name.startswith(self.name)]
        self.num_vars = count_model_params(tf_variables)
        print("Model has {0} params".format(self.num_vars))

        self.train_op, self.lr, self.grad_norm, self.optimizer, self.grads = get_train_ops(
            train_loss,
            tf_variables,
            self.global_step,
            clip_mode=self.clip_mode,
            grad_bound=self.grad_bound,
            l2_reg=self.l2_reg,
            lr_init=self.lr_init,
            lr_dec_start=self.lr_dec_start,
            lr_warmup_steps = self.lr_warmup_steps,
            lr_warmup_val = self.lr_warmup_val,            
            lr_dec_rate=self.lr_dec_rate,
            lr_dec_every=self.lr_dec_every,
            lr_dec_min=self.lr_dec_min,
            lr_cosine=self.lr_cosine,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            lr_T_0=self.lr_T_0,
            lr_T_mul=self.lr_T_mul,
            num_train_batches=self.num_train_batches,
            optim_algo=self.optim_algo
            # sync_replicas=self.sync_replicas,
            # num_aggregate=self.num_aggregate,
            # num_replicas=self.num_replicas
        )

    # override
    def _build_valid(self):
        if self.x_valid is not None:
            print("-" * 80)
            print("Build valid graph")
            self.valid_preds = self._model(self.x_valid, False, reuse=True)
            # self.loss = tf.losses.mean_squared_error(labels=self.y_valid, predictions=self.valid_preds)
            # self.valid_PSNR = tf.image.psnr(self.y_valid, preds, 1)
            # self.valid_PSNR = tf.reduce_sum(self.valid_PSNR)

    # override
    def _build_test(self):
        print("-" * 80)
        print("Build test graph")
        self.test_preds = self._model(self.x_test, False, reuse=True)
        # self.loss = tf.losses.mean_squared_error(labels=self.y_test, predictions=self.test_preds)
        # self.test_PSNR = tf.image.psnr(self.y_test, self.test_preds, 1)
        # self.test_PSNR = tf.reduce_sum(self.test_PSNR)

    # override
    def build_valid_rl(self, shuffle=False):
        if self.x_valid_rl is not None:
            print("-" * 80)
            print("Build valid graph for rl")
            if self.use_model == "SRCNN":
                self.valid_preds_rl = self._model_srcnn(self.x_valid_rl, False, reuse=True)
            elif self.use_model == "RDN":
                self.valid_preds_rl = self._model_RDN(self.x_valid_rl, False, reuse=True)
            elif self.use_model == "CARN":
                self.valid_preds_rl = self._model_CARN(self.x_valid_rl, False, reuse=True)

            else:
                self.valid_preds_rl = self._model(self.x_valid_rl, False, reuse=True)
            self.valid_rl_PSNR = tf.Variable(0.,dtype=tf.float32)
            # self.loss = tf.losses.mean_squared_error(labels=self.y_valid_rl, predictions=self.valid_preds_rl)
            # self.valid_rl_PSNR = tf.image.psnr(self.y_valid_rl, self.valid_preds_rl, 1)
            # self.valid_rl_PSNR = tf.reduce_sum(self.valid_rl_PSNR)

    def connect_controller(self, controller_model):
        if self.exist_fixed_arc:
            self.fixed_arc_Low = np.array([int(x) for x in self.fixed_arcs[0].split(" ") if x])
            self.fixed_arc_Mid = np.array([int(x) for x in self.fixed_arcs[1].split(" ") if x])
            self.fixed_arc_High = np.array([int(x) for x in self.fixed_arcs[2].split(" ") if x])
            self.fixed_arcs = []
            self.fixed_arcs.append(self.fixed_arc_Low)
            self.fixed_arcs.append(self.fixed_arc_Mid)
            self.fixed_arcs.append(self.fixed_arc_High)
        else:
            # self.fixed_arc = controller_model.sample_arc
            self.now_arc_Low = tf.placeholder(dtype=tf.int32)
            self.now_arc_Mid = tf.placeholder(dtype=tf.int32)
            self.now_arc_High = tf.placeholder(dtype=tf.int32)
            self.fixed_arcs = []
            self.fixed_arcs.append(self.now_arc_Low)
            self.fixed_arcs.append(self.now_arc_Mid)
            self.fixed_arcs.append(self.now_arc_High)
            # self.fixed_arcs = tf.Print(self.fixed_arcs, [self.fixed_arcs], message="now arc is:",summarize=-1)

        self._build_train()
        self._build_valid()
        self._build_test()
