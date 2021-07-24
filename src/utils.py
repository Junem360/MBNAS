import sys
import numpy as np
import tensorflow as tf

user_flags = []


def DEFINE_string(name, default_value, doc_string):
    tf.app.flags.DEFINE_string(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)


def DEFINE_integer(name, default_value, doc_string):
    tf.app.flags.DEFINE_integer(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)


def DEFINE_float(name, default_value, doc_string):
    tf.app.flags.DEFINE_float(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)


def DEFINE_boolean(name, default_value, doc_string):
    tf.app.flags.DEFINE_boolean(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)


def print_user_flags(line_limit=80):
    print("-" * 80)

    global user_flags
    FLAGS = tf.app.flags.FLAGS

    for flag_name in sorted(user_flags):
        value = "{}".format(getattr(FLAGS, flag_name))
        log_string = flag_name
        log_string += "." * (line_limit - len(flag_name) - len(value))
        log_string += value
        print(log_string)


class Logger(object):
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()


def count_model_params(tf_variables):
    """
    Args:
        tf_variables: list of all model variables
    """

    num_vars = 0
    for var in tf_variables:
        num_vars += np.prod([dim.value for dim in var.get_shape()])
    return num_vars


def get_train_ops(
        loss,
        tf_variables,
        train_step,
        clip_mode=None,
        global_error=0,
        grad_bound=None,
        l2_reg=1e-4,
        lr_warmup_val=None,
        lr_warmup_steps=100,
        lr_init=1e-4,
        lr_dec_start=0,
        lr_dec_every=10000,
        lr_dec_rate=0.1,
        lr_dec_min=None,
        lr_cosine=False,
        lr_max=None,
        lr_min=None,
        lr_T_0=None,
        lr_T_mul=None,
        num_train_batches=None,
        optim_algo=None,
        sync_replicas=False,
        num_aggregate=None,
        num_replicas=None,
        get_grad_norms=False,
        moving_average=None):
    """
    Args:
        clip_mode: "global", "norm", or None.
        moving_average: store the moving average of parameters
    """

    if l2_reg > 0:
        l2_losses = []
        for var in tf_variables:
            l2_losses.append(tf.reduce_sum(var ** 2))
        l2_loss = tf.add_n(l2_losses)
        loss += l2_reg * l2_loss

    grads = tf.gradients(loss, tf_variables)

    # grads = tf.Print(grads, [grads[0]], message="grads :", summarize=200)

    grad_norm = tf.global_norm(grads)
    # grad_norm = tf.minimum(tf.float32.max, tf.global_norm(grads))
    # grad_norm = tf.Print(grad_norm, [grad_norm], message="grad_norm :",summarize=200)
    grad_norms = {}
    for v, g in zip(tf_variables, grads):
        if v is None or g is None:
            continue
        if isinstance(g, tf.IndexedSlices):
            grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g.values ** 2))
        else:
            grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g ** 2))

    if clip_mode is not None:
        clipped = []
        for g in grads:

            c_g = tf.clip_by_value(g, tf.float32.min, tf.float32.max)
            clipped.append(c_g)
        grads = clipped
        norm_global = tf.minimum(tf.float32.max, tf.global_norm(grads))
        assert grad_bound is not None, "Need grad_bound to clip gradients."
        if clip_mode == "global":
            grads, _ = tf.clip_by_global_norm(grads, grad_bound, use_norm=norm_global)

        elif clip_mode == "norm":
            clipped = []
            for g in grads:
                if isinstance(g, tf.IndexedSlices):
                    c_g = tf.clip_by_norm(g.values, grad_bound)
                    c_g = tf.IndexedSlices(g.indices, c_g)
                else:
                    c_g = tf.clip_by_norm(g, grad_bound)
                clipped.append(c_g)
            grads = clipped
        else:
            raise NotImplementedError("Unknown clip_mode {}".format(clip_mode))



    learning_rate = tf.train.exponential_decay(
        lr_init, tf.maximum(train_step - lr_dec_start, 0), lr_dec_every,
        lr_dec_rate, staircase=True)
    if lr_dec_min is not None:
        learning_rate = tf.maximum(learning_rate, lr_dec_min)

    if lr_warmup_val is not None:
        learning_rate = tf.cond(tf.less(train_step, lr_warmup_steps),
                                lambda: lr_warmup_val, lambda: learning_rate)

    if get_grad_norms:
        g_1, g_2 = 0.0001, 0.0001
        for v, g in zip(tf_variables, grads):
            if g is not None:
                if isinstance(g, tf.IndexedSlices):
                    g_n = tf.reduce_sum(g.values ** 2)
                else:
                    g_n = tf.reduce_sum(g ** 2)
                if "enas_cell" in v.name:
                    print("g_1: {}".format(v.name))
                    g_1 += g_n
                else:
                    print("g_2: {}".format(v.name))
                    g_2 += g_n
        learning_rate = tf.Print(learning_rate, [g_1, g_2, tf.sqrt(g_1 / g_2)],
                                 message="g_1, g_2, g_1/g_2: ", summarize=5)

    if optim_algo == "momentum":
        opt = tf.train.MomentumOptimizer(
            learning_rate, 0.9, use_locking=True, use_nesterov=True)
    elif optim_algo == "sgd":
        opt = tf.train.GradientDescentOptimizer(learning_rate, use_locking=True)
    elif optim_algo == "adam":
        opt = tf.train.AdamOptimizer(learning_rate, use_locking=True)
    elif optim_algo == "RMSprop":
        opt = tf.train.RMSPropOptimizer(learning_rate, use_locking=True)
    else:
        raise ValueError("Unknown optim_algo {}".format(optim_algo))

    if sync_replicas:
        assert num_aggregate is not None, "Need num_aggregate to sync."
        assert num_replicas is not None, "Need num_replicas to sync."

        opt = tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=num_aggregate,
            total_num_replicas=num_replicas,
            use_locking=True)

    if moving_average is not None:
        opt = tf.contrib.opt.MovingAverageOptimizer(
            opt, average_decay=moving_average)

    train_op = opt.apply_gradients(
        zip(grads, tf_variables), global_step=train_step)  # add +1 to global step

    if get_grad_norms:
        return train_op, learning_rate, grad_norm, opt, grad_norms, grads
    else:
        return train_op, learning_rate, grad_norm, opt, grads



def calculate_cb_penalty(now_arc, num_cells, num_layers):
    complex_model = 4*num_cells
    now_model_matrix_Low = np.zeros((3,2*num_cells))
    for i in range(len(now_arc[0])):
        for j in range(3):
            now_model_matrix_Low[now_arc[j][i]][i] = 1
    now_model_Low = (np.sum(now_model_matrix_Low)-2*num_cells)

    cb_reward = (now_model_Low) / complex_model

    return cb_reward