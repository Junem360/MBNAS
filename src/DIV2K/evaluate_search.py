import os
import shutil
import sys
import time
import glob
import pickle

import numpy as np
import tensorflow as tf
import cv2
from scipy import signal
from scipy import ndimage

from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import print_user_flags
from src.utils import calculate_cb_penalty

from src.utils_img import imresize
from src.utils_img import bgr2y
from src.utils_img import calc_psnr
from src.utils_img import modcrop
from src.utils_img import calculate_psnr
from src.utils_img import calculate_ssim

from src.DIV2K.data_utils import make_eval_batch_lr
from src.DIV2K.data_utils import make_eval_batch
from src.DIV2K.data_utils import get_random_batch_early
from src.DIV2K.data_utils import DataLoader
from src.DIV2K.controller import ControllerNetwork
from src.DIV2K.child_network import ChildNetwork

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

flags = tf.app.flags
FLAGS = flags.FLAGS
# parameters for result, in/out data
DEFINE_boolean("reset_output_dir", True, "Delete output_dir if exists.")
DEFINE_string("data_path", "../tfrecord/", "path of train,valid tfrecord folder")
DEFINE_string("img_path", "../data/", "path of test image folder")
DEFINE_string("data_format", "NHWC", "image data format. 'NHWC' or 'NCHW'")
DEFINE_string("output_dir", "./outputs/x2_evaluate_search", "path of result")
DEFINE_string("checkpoint", "./outputs/x2_search_2/model.ckpt-500000", "path of checkpoint file")

DEFINE_boolean("test_mode", False, "use when test")
DEFINE_boolean("inference_mode", False, "use when inference")
DEFINE_string("use_model", None, "which model to use for training")
DEFINE_boolean("rl_search", False, "use global/local feature fusion searching")
DEFINE_boolean("cb_reward", True, "use complexity based reward")
DEFINE_float("cb_rate", 2, "rate of complexity based reward")

# parameters for batch and training
DEFINE_integer("batch_size", 16, "batch size in training process")
DEFINE_integer("eval_batch_size", 10, "batch size of evaluation process")
DEFINE_integer("test_batch_size", 1, "batch size of test process")
DEFINE_integer("patch_size", 36, "patch size in training process")
DEFINE_integer("eval_patch_size", 48, "patch size in training process")
DEFINE_integer("num_epochs", 500, "training epoch for child_network")
DEFINE_integer("it_per_epoch", 1000, "iteration of 1 epoch for child_network")

DEFINE_integer("loss_cut", 5, "cut training process when loss > avgLoss*loss_cut")
# DEFINE_boolean("image_random", True, "use when test")
DEFINE_boolean("fine_tune", False, "use when test")
DEFINE_boolean("channel_attn", True, "use channel_attn method or not")

# parameters for child_network design
DEFINE_integer("child_upsample_size", 2, "rate of lr image size")
DEFINE_integer("child_num_layers", 4, "number of Cells")
DEFINE_integer("child_num_cells", 2, "number of layers in cells")
DEFINE_integer("child_out_filters", 32, "number of out filter channels of each cells")
DEFINE_integer("child_num_sfe", 32, "number of out filter channels of shallow feature extraction layer")
DEFINE_integer("child_num_branches", 3, "number of operations in search space")

DEFINE_string("child_fixed_arc_Low", None, "")
DEFINE_string("child_fixed_arc_Mid", None, "")
DEFINE_string("child_fixed_arc_High", None, "")

DEFINE_boolean("child_use_aux_heads", False, "Should we use an aux head")
DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")
DEFINE_integer("child_num_aggregate", None, "")
DEFINE_integer("child_num_replicas", 1, "")

# DEFINE_integer("child_block_size", 3, "")
# DEFINE_integer("child_out_filters_scale", 1, "")
# DEFINE_integer("child_filter_size", 5, ""

# parameters for child_network learning rate, gradient, loss
DEFINE_integer("child_lr_dec_every", 100, "learning rate decay step size of child network")
DEFINE_integer("child_cutout_size", None, "CutOut size")
DEFINE_float("child_grad_bound", None, "Gradient clipping")
DEFINE_float("child_lr", 3*1e-4, "")
DEFINE_float("child_lr_dec_rate", 0.5, "")
DEFINE_float("child_lr_dec_min", 1e-12, "")
DEFINE_float("child_l2_reg", 0, "")
DEFINE_float("child_lr_warmup_val", None, "warming up learning rate")
DEFINE_integer("child_lr_warmup_step", 0, "step to use warmup learning rate")
DEFINE_integer("child_lr_dec_start", 0, "step to start learning rate decrease")
# parameters for child_network cosine lr
DEFINE_boolean("child_lr_cosine", False, "Use cosine lr schedule")
DEFINE_float("child_lr_max", 0.05, "for lr schedule")
DEFINE_float("child_lr_min", 0.0005, "for lr schedule")
DEFINE_integer("child_lr_T_0", 10, "for lr schedule")
DEFINE_integer("child_lr_T_mul", 2, "for lr schedule")

# parameters for controller
DEFINE_float("controller_lr", 0.0003, "")
DEFINE_float("controller_lr_dec_rate", 0.8, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", 1.10, "")
DEFINE_float("controller_op_tanh_reduce", 2.5, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_float("controller_entropy_weight", 0.0001, "")
DEFINE_float("controller_skip_target", 0.8, "")
DEFINE_float("controller_skip_weight", 0.0, "")
DEFINE_integer("controller_num_aggregate", None, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 100, "")
DEFINE_integer("controller_train_every", 1, "train the controller after this number of epochs")
DEFINE_integer("controller_train_start", 5, "start controller training epoch")
DEFINE_float("controller_best_rate", 0, "rate of training controller by best architecture")

DEFINE_boolean("controller_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("controller_training", True, "")

DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_integer("controller_forwards_limit", 2, "")
DEFINE_boolean("controller_use_critic", False, "")

DEFINE_integer("log_every", 200, "How many steps to log")
DEFINE_integer("controller_log_every", 20, "How many steps to log when training controller")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

# parameters for ???
# DEFINE_float("child_keep_prob", 0.90, "")
# DEFINE_float("child_drop_path_keep_prob", 0.60, "minimum drop_path_keep_prob")
DEFINE_string("child_skip_pattern", None, "Must be ['dense', None]")

def evaluate():
    fixed_arcs = None
    images, labels, meta_data = {}, {}, {}
    test_set = ['Set5']
    # test_set = ['Set5', 'Set14', 'BSDS100', 'Urban100']
    labels["valid"] = None
    labels["valid_rl"] = None
    test_file_paths = []
    for i in range(len(test_set)):
        test_file_paths.append(os.path.join(FLAGS.img_path, test_set[i]))
    test_file_path = os.path.join(FLAGS.img_path, 'Set5')
    perf_value = 0
    g = tf.Graph()
    with g.as_default():

        for i in range(len(test_set)):
            images[test_set[i]], labels[test_set[i]], meta_data[test_set[i]] = make_eval_batch_lr(test_file_paths[i],
                                                        'x2_small', FLAGS.child_upsample_size)
            print("data_num of {} = {}".format(test_set[i], meta_data[test_set[i]]["total_data_num"]))
        images["test"], labels["test"], meta_data["test"] = make_eval_batch(test_file_path, 'x2_small',
                                                                            FLAGS.child_upsample_size)


        print("build controller, child_network...")
        controllerClass = ControllerNetwork
        childClass = ChildNetwork
        child_model = childClass(
            images,
            labels,
            meta_data,
            output_dir=FLAGS.output_dir,
            use_aux_heads=FLAGS.child_use_aux_heads,
            use_model=FLAGS.use_model,
            fine_tune=FLAGS.fine_tune,
            channel_attn=FLAGS.channel_attn,
            cb_reward=FLAGS.cb_reward,
            cutout_size=FLAGS.child_cutout_size,
            num_layers=FLAGS.child_num_layers,
            num_cells=FLAGS.child_num_cells,
            num_branches=FLAGS.child_num_branches,
            fixed_arcs=fixed_arcs,
            out_filters=FLAGS.child_out_filters,
            upsample_size=FLAGS.child_upsample_size,
            sfe_filters=FLAGS.child_num_sfe,
            num_epochs=FLAGS.num_epochs,
            it_per_epoch=FLAGS.it_per_epoch,
            l2_reg=FLAGS.child_l2_reg,
            data_format=FLAGS.data_format,
            batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            test_batch_size=FLAGS.test_batch_size,
            clip_mode=None,
            grad_bound=FLAGS.child_grad_bound,
            lr_init=FLAGS.child_lr,
            lr_dec_every=FLAGS.child_lr_dec_every,
            lr_dec_rate=FLAGS.child_lr_dec_rate,
            lr_dec_min=FLAGS.child_lr_dec_min,
            lr_warmup_val=FLAGS.child_lr_warmup_val,
            lr_warmup_step=FLAGS.child_lr_warmup_step,
            lr_dec_start=FLAGS.child_lr_dec_start,
            lr_cosine=FLAGS.child_lr_cosine,
            lr_max=FLAGS.child_lr_max,
            lr_min=FLAGS.child_lr_min,
            lr_T_0=FLAGS.child_lr_T_0,
            lr_T_mul=FLAGS.child_lr_T_mul,
            optim_algo="adam",
            sync_replicas=FLAGS.child_sync_replicas,
            num_aggregate=FLAGS.child_num_aggregate,
            num_replicas=FLAGS.child_num_replicas,
        )

        if FLAGS.child_fixed_arc_Low is None:
            print("fixed arc is None. training controllers.")
            controller_model = controllerClass(
                use_cb_reward=FLAGS.cb_reward,
                cb_rate=FLAGS.cb_rate,
                skip_target=FLAGS.controller_skip_target,
                skip_weight=FLAGS.controller_skip_weight,
                num_cells=FLAGS.child_num_cells,
                num_layers=FLAGS.child_num_layers,
                num_branches=FLAGS.child_num_branches,
                out_filters=FLAGS.child_out_filters,
                lstm_size=64,
                lstm_num_layers=2,
                lstm_keep_prob=1.0,
                tanh_constant=FLAGS.controller_tanh_constant,
                op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
                temperature=FLAGS.controller_temperature,
                lr_init=FLAGS.controller_lr,
                lr_dec_start=0,
                lr_dec_every=1000000,  # never decrease learning rate
                l2_reg=FLAGS.controller_l2_reg,
                entropy_weight=FLAGS.controller_entropy_weight,
                bl_dec=FLAGS.controller_bl_dec,
                use_critic=FLAGS.controller_use_critic,
                optim_algo="adam",
                sync_replicas=FLAGS.controller_sync_replicas,
                num_aggregate=FLAGS.controller_num_aggregate,
                num_replicas=FLAGS.controller_num_replicas)

            child_model.connect_controller(controller_model)
            controller_model.build_trainer(child_model)

            controller_ops = {
                "train_step": controller_model.train_step,
                "loss": controller_model.loss,
                "train_op": controller_model.train_op,
                "lr": controller_model.lr,
                "grad_norm": controller_model.grad_norm,
                "valid_PSNR": controller_model.valid_PSNR,
                "optimizer": controller_model.optimizer,
                "baseline": controller_model.baseline,
                "entropy": controller_model.sample_entropy,
                "sample_arc": controller_model.sample_arc,
                "skip_rate": controller_model.skip_rate,
                "batch_size": child_model.batch_size,
                "reward": controller_model.reward,
                "log_prob": controller_model.sample_log_prob,
            }
        else:
            # print(fixed_arcs)
            assert not FLAGS.controller_training, (
                "--child_fixed_arc is given, cannot train controller")
            child_model.connect_controller(None)
            controller_ops = None


        with tf.Session() as sess:

            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(sess=sess, coord=coord)
            restorer = tf.train.Saver(max_to_keep=2)

            sess.run(tf.global_variables_initializer())

            # ckpt_state = tf.train.get_checkpoint_state("outputs/search_result")

            restorer.restore(sess, FLAGS.checkpoint)

            print("Here are 500 architectures")
            arc_data = []
            PSNR_data = []
            penalty_data = []
            reward_data = []

            num_examples = meta_data[test_set[0]]['total_data_num']
            num_batches = num_examples
            pred_img_op = child_model.test_preds
            test_img = images[test_set[0]]
            test_label = labels[test_set[0]]

            output_dir = child_model.output_dir
            bgr_mean = np.array([103.154 / 255, 111.561 / 255, 114.356 / 255]).astype(np.float32)

            for _ in range(500):
                now_arcs = sess.run(controller_model.sample_arc)
                total_PSNR = 0

                for batch_id in range(num_batches):
                    h_i, w_i, c_i = test_img[batch_id].shape
                    h_o, w_o, c_o = test_label[batch_id].shape
                    inp_test = test_img[batch_id] - bgr_mean
                    test_data = np.reshape(inp_test, [1, h_i, w_i, c_i])

                    pred_img = sess.run(pred_img_op, feed_dict={child_model.now_arc_Low: now_arcs[0],
                                                                child_model.now_arc_Mid: now_arcs[1],
                                                                child_model.now_arc_High: now_arcs[2],
                                                                child_model.x_test: test_data})
                    pred_img = np.reshape(pred_img, [h_o, w_o, c_o]) + bgr_mean
                    pred_img = (np.round(np.clip(pred_img * 255., 0., 255.)) / 255).astype(np.float32)

                    input_img = test_img[batch_id]
                    label_img = test_label[batch_id]
                    # cv2.imwrite("./result_img/{}/{}".format(test_set[i],
                    #                                         meta_data[test_set[i]]["img_names"][batch_id]),
                    #             pred_img * 255)
                    pred_img = pred_img[2:-2, 2:-2, :]
                    input_img = input_img[2:-2, 2:-2, :]
                    label_img = label_img[2:-2, 2:-2, :]

                    pred_img_y = bgr2y(pred_img)
                    label_img_y = bgr2y(label_img)

                    PSNR = calc_psnr(pred_img_y, label_img_y)
                    total_PSNR += PSNR
                    # print("image_{}'s PSNR = {}, SSIM = {}".format(batch_id, PSNR, SSIM))
                total_exp = num_examples
                psnr = total_PSNR / total_exp
                print("{} database, test_PSNR={:<6.4f}".format(test_set[i], psnr))





                cb_reward = calculate_cb_penalty(now_arcs,child_model.num_cells, child_model.num_layers)
                cb_reward = cb_reward
                # label_data, result_data = sess.run(run_ops, {child_model.now_arc: now_arc})
                # val_psnr = calculate_psnr(result_data, label_data)

                print(np.reshape(now_arcs, [-1]))
                arc_data.append(np.reshape(now_arcs, [-1]))
                PSNR_data.append(psnr)
                penalty_data.append(cb_reward)
                reward_data.append(psnr-cb_reward)
                print("val_PSNR={:<6.4f}, cb_penalty={:<6.4f}, reward={:<6.4f}".format(psnr, cb_reward,
                                                                                       psnr - cb_reward))
                print("-" * 80)

                np.savetxt("arc_data.csv", arc_data, delimiter=",", fmt="%d")
                np.savetxt("PSNR_data.csv", PSNR_data, delimiter=",", fmt="%6.4f")
                np.savetxt("penalty_data.csv", penalty_data, delimiter=",", fmt="%6.4f")
                np.savetxt("reward_data.csv", reward_data, delimiter=",", fmt="%6.4f")



def main(_):
    print("-" * 80)
    if not os.path.isdir(FLAGS.output_dir):
        print(FLAGS.output_dir)
        print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
        os.makedirs(FLAGS.output_dir)
    elif FLAGS.reset_output_dir:
        print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
        shutil.rmtree(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)
                
    utils.print_user_flags()
    evaluate()


if __name__ == "__main__":
    tf.app.run()
