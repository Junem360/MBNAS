import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

from src.ops_general import create_weight
from src.ops_general import create_bias


def Channel_attention(x, ratio, num_features, data_format):

    residual = x

    x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    w_d = create_weight("w_d", [1, 1, num_features, num_features // ratio])
    b_d = create_weight("b_d", [num_features // ratio])
    x_d = tf.nn.conv2d(
        x, w_d, [1, 1, 1, 1], "SAME", data_format=data_format) + b_d
    x_d = tf.nn.relu(x_d)

    w_u = create_weight("w_u", [1, 1, num_features // ratio, num_features])
    b_u = create_weight("b_u", [num_features])
    x_u = tf.nn.conv2d(
        x_d, w_u, [1, 1, 1, 1], "SAME", data_format=data_format) + b_u

    x = tf.nn.sigmoid(x_u)
    x = tf.multiply(x, residual)

    return x

def Pixel_shuffler(x, num_features, out_features, data_format, upsample_size):

    if upsample_size == 4:
        w1 = create_weight("w_ps1", [3, 3, num_features , out_features*2*2])
        b1 = create_weight("b_ps1", [out_features*2*2])
        x = tf.nn.conv2d(
            x, w1, [1, 1, 1, 1], "SAME", data_format=data_format) + b1
        x = tf.depth_to_space(x, 2)
        w2 = create_weight("w_ps2", [3, 3, out_features , out_features*2*2])
        b2 = create_weight("b_ps2", [out_features*2*2])
        x = tf.nn.conv2d(
            x, w2, [1, 1, 1, 1], "SAME", data_format=data_format) + b2
        x = tf.depth_to_space(x, 2)

    else:
        w = create_weight("w_ps", [3, 3, num_features , out_features*upsample_size*upsample_size])
        b = create_weight("b_ps", [out_features*upsample_size*upsample_size])
        x = tf.nn.conv2d(
            x, w, [1, 1, 1, 1], "SAME", data_format=data_format) + b
        x = tf.depth_to_space(x, upsample_size)

    return x

def Texture_attention(x, texture, num_features, data_format):

    w1 = create_weight("w_ta1", [1, 1, 3, num_features])
    b1 = create_weight("b_ta1", [num_features])
    tex_attn = tf.nn.conv2d(
        texture, w1, [1, 1, 1, 1], "SAME", data_format=data_format) + b1
    tex_attn = tf.nn.relu(tex_attn)
    w2 = create_weight("w_ta2", [3, 3, num_features, num_features])
    b2 = create_weight("b_ta2", [num_features])
    tex_attn = tf.nn.conv2d(
        tex_attn, w2, [1, 1, 1, 1], "SAME", data_format=data_format) + b2
    tex_attn = tf.nn.relu(tex_attn)
    w3 = create_weight("w_ta3", [3, 3, num_features, num_features])
    b3 = create_weight("b_ta3", [num_features])
    tex_attn = tf.nn.conv2d(
        tex_attn, w3, [1, 1, 1, 1], "SAME", data_format=data_format) + b3

    out = tf.multiply(x,tex_attn)

    return out