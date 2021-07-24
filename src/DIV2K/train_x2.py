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
from src.utils_img import get_texture_map

from src.DIV2K.data_utils import threaded_input_word_pipeline_texture
from src.DIV2K.data_utils import make_eval_batch_texture_lr
from src.DIV2K.data_utils import make_eval_batch
from src.DIV2K.data_utils import get_random_batch_texture_early
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
DEFINE_string("texture_path", "../texture_map/", "path of texture_map folder")
DEFINE_string("texture_size", "31", "size of texture filter")
DEFINE_string("texture_lr_path", "../texture_image/x2_{}".format(FLAGS.texture_size), "path of texture_map folder")
DEFINE_string("img_path", "../data/", "path of test image folder")
DEFINE_string("data_format", "NHWC", "image data format. 'NHWC' or 'NCHW'")
DEFINE_string("output_dir", "./outputs/x2_train", "path of result")
DEFINE_string("checkpoint", "./outputs/x2/model.ckpt-931000", "path of checkpoint file")
DEFINE_string("texture_checkpoint", "./outputs/texture_map_x2/model.ckpt-500000",
              "path of texture checkpoint file")

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
DEFINE_integer("num_epochs", 1000, "training epoch for child_network")
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

DEFINE_string("child_fixed_arc_Low", "0 0 1 2", "")
DEFINE_string("child_fixed_arc_Mid", "0 0 1 2", "")
DEFINE_string("child_fixed_arc_High", "2 0 1 2", "")
DEFINE_string("texture_arc", "1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0", "")

DEFINE_boolean("child_use_aux_heads", False, "Should we use an aux head")
DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")
DEFINE_integer("child_num_aggregate", None, "")
DEFINE_integer("child_num_replicas", 1, "")

# DEFINE_integer("child_block_size", 3, "")
# DEFINE_integer("child_out_filters_scale", 1, "")
# DEFINE_integer("child_filter_size", 5, ""

# parameters for child_network learning rate, gradient, loss
DEFINE_integer("child_lr_dec_every", 200, "learning rate decay step size of child network")
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
DEFINE_float("controller_lr_dec_rate", 1.0, "")
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
DEFINE_boolean("controller_training", False, "")

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


def train():
    print("train func...")
    print("get data...")
    if (FLAGS.child_fixed_arc_Low is None):
        fixed_arcs = None
    else:
        fixed_arcs = []
        fixed_arcs.append(FLAGS.child_fixed_arc_Low)
        fixed_arcs.append(FLAGS.child_fixed_arc_Mid)
        fixed_arcs.append(FLAGS.child_fixed_arc_High)

    images, labels, meta_data, texture_maps = {}, {}, {}, {}
    train_img_path = os.path.join(FLAGS.img_path, 'DIV2K_train_HR/*.png')
    # train_file_path = os.path.join(FLAGS.data_path, 'DIV2K_train_x2_texture')
    valid_img_path = os.path.join(FLAGS.img_path, 'DIV2K_valid_HR/*.png')
    valid_file_path = os.path.join(FLAGS.data_path, 'DIV2K_valid_x2_texture')
    texture_file_path = os.path.join(FLAGS.texture_lr_path, 'DIV2K_train_HR')
    valid_texture_file_path = os.path.join(FLAGS.texture_lr_path, 'DIV2K_valid_HR')
    test_set = ['Set5', 'Set14', 'BSDS100', 'Urban100']
    test_file_paths = []
    test_texture_paths = []
    for i in range(4):
        test_file_paths.append(os.path.join(FLAGS.img_path, test_set[i]))
        test_texture_paths.append(os.path.join(FLAGS.texture_lr_path, test_set[i]))
    test_file_path = os.path.join(FLAGS.img_path, 'Set5')
    perf_value = 0
    g = tf.Graph()
    with g.as_default():
        lTrainImgName = glob.glob(train_img_path)
        nImg = len(lTrainImgName)
        train_images = []
        train_textures = []
        # for i in range(10):
        for i in range(nImg):
            tmp_texture = []
            tmp_name = lTrainImgName[i]
            img = cv2.imread(lTrainImgName[i])
            train_images.append(img)

            texture_file_name_Low = "{}{}".format(tmp_name[tmp_name.rfind('/') + 1:-4],
                                                  "_Low_{}.png".format(FLAGS.texture_size))
            tmp_name_Low = os.path.join(texture_file_path, "Low", texture_file_name_Low)
            tmp_texture_Low = cv2.imread(tmp_name_Low)

            texture_file_name_Mid = "{}{}".format(tmp_name[tmp_name.rfind('/') + 1:-4],
                                                  "_Mid_{}.png".format(FLAGS.texture_size))
            tmp_name_Mid = os.path.join(texture_file_path, "Mid", texture_file_name_Mid)
            tmp_texture_Mid = cv2.imread(tmp_name_Mid)

            texture_file_name_High = "{}{}".format(tmp_name[tmp_name.rfind('/') + 1:-4],
                                                   "_High_{}.png".format(FLAGS.texture_size))
            tmp_name_High = os.path.join(texture_file_path, "High", texture_file_name_High)
            tmp_texture_High = cv2.imread(tmp_name_High)

            tmp_texture.append(tmp_texture_Low)
            tmp_texture.append(tmp_texture_Mid)
            tmp_texture.append(tmp_texture_High)

            train_textures.append(tmp_texture)

        labels["train"] = train_images
        texture_maps["train"] = train_textures

        lValidImgName = glob.glob(valid_img_path)
        nImg = len(lValidImgName)
        valid_images = []
        valid_textures = []
        # for i in range(10):
        for i in range(nImg):
            tmp_texture = []
            tmp_name = lValidImgName[i]
            img = cv2.imread(lValidImgName[i])
            valid_images.append(img)

            texture_file_name_Low = "{}{}".format(tmp_name[tmp_name.rfind('/') + 1:-4],
                                                  "_Low_{}.png".format(FLAGS.texture_size))
            tmp_name_Low = os.path.join(valid_texture_file_path, "Low", texture_file_name_Low)
            tmp_texture_Low = cv2.imread(tmp_name_Low)

            texture_file_name_Mid = "{}{}".format(tmp_name[tmp_name.rfind('/') + 1:-4],
                                                  "_Mid_{}.png".format(FLAGS.texture_size))
            tmp_name_Mid = os.path.join(valid_texture_file_path, "Mid", texture_file_name_Mid)
            tmp_texture_Mid = cv2.imread(tmp_name_Mid)

            texture_file_name_High = "{}{}".format(tmp_name[tmp_name.rfind('/') + 1:-4],
                                                   "_High_{}.png".format(FLAGS.texture_size))
            tmp_name_High = os.path.join(valid_texture_file_path, "High", texture_file_name_High)
            tmp_texture_High = cv2.imread(tmp_name_High)

            tmp_texture.append(tmp_texture_Low)
            tmp_texture.append(tmp_texture_Mid)
            tmp_texture.append(tmp_texture_High)

            valid_textures.append(tmp_texture)

        labels["valid"] = valid_images
        labels["valid_rl"] = valid_images
        texture_maps["valid"] = valid_textures

        train_meta = {}
        train_meta["data_nums"] = 1000000
        train_meta["total_data_num"] = 1000000
        train_meta["file_num"] = 1
        train_meta["data_files"] = 1
        meta_data["train"] = train_meta
        valid_meta = {}
        valid_meta["data_nums"] = 1000
        valid_meta["total_data_num"] = 1000
        valid_meta["file_num"] = 1
        valid_meta["data_files"] = 1
        meta_data["valid"] = valid_meta
        meta_data["valid_rl"] = valid_meta

        # images["train"], labels["train"], texture_maps["train"] = threaded_input_word_pipeline_texture(train_file_path,
        #                                                     file_patterns=[
        #                                                         '*.tfrecord'],
        #                                                     num_threads=4,
        #                                                     batch_size=FLAGS.batch_size,
        #                                                     img_size=FLAGS.patch_size,
        #                                                     label_size=FLAGS.patch_size * FLAGS.child_upsample_size,
        #                                                     num_epochs=None,
        #                                                     is_train=False)

        # images["valid"], labels["valid"], texture_maps["valid"] = threaded_input_word_pipeline_texture2(valid_file_path,
        #                                                                                     file_patterns=[
        #                                                                                         '*.tfrecord'],
        #                                                                                     num_threads=4,
        #                                                                                     batch_size=FLAGS.eval_batch_size,
        #                                                                                     img_size=FLAGS.eval_patch_size,
        #                                                                                     label_size=FLAGS.eval_patch_size*FLAGS.child_upsample_size,
        #                                                                                     num_epochs=None,
        #                                                                                     is_train=False)
        for i in range(4):
            images[test_set[i]], labels[test_set[i]], texture_maps[test_set[i]], meta_data[
                test_set[i]] = make_eval_batch_texture_lr(test_file_paths[i], test_texture_paths[i], 'x2_small',
                                                        FLAGS.child_upsample_size, FLAGS.texture_size)
            print("data_num of {} = {}".format(test_set[i], meta_data[test_set[i]]["total_data_num"]))
        images["test"], labels["test"], meta_data["test"] = make_eval_batch(test_file_path, 'x2_small',
                                                                            FLAGS.child_upsample_size)

        # if FLAGS.child_fixed_arc_Low is None:
        #     images["valid_rl"], labels["valid_rl"], texture_maps["valid_rl"] = threaded_input_word_pipeline_texture2(
        #         valid_file_path,
        #         file_patterns=[
        #             '*.tfrecord'],
        #         num_threads=4,
        #         batch_size=FLAGS.eval_batch_size,
        #         img_size=FLAGS.eval_patch_size,
        #         label_size=FLAGS.eval_patch_size*FLAGS.child_upsample_size,
        #         num_epochs=None,
        #         is_train=False)
        # else:
        #     images["valid_rl"], labels["valid_rl"], meta_data["valid_rl"] = None, None, None

        if labels["valid"] is not None:
            print("data_num of train data = {}".format(meta_data["train"]["total_data_num"]))
            # print("data_num of test data = {}".format(meta_data["test"]["total_data_num"]))
            print("data_num of valid data = {}".format(meta_data["valid"]["total_data_num"]))
        else:
            print("data_num of train data = {}".format(meta_data["train"]["total_data_num"]))
            # print("data_num of test data = {}".format(meta_data["test"]["total_data_num"]))

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
        child_ops = {
            "global_step": child_model.global_step,
            "loss": child_model.loss,
            "train_op": child_model.train_op,
            "lr": child_model.lr,
            "grads": child_model.grads,
            "grad_norm": child_model.grad_norm,
            "optimizer": child_model.optimizer,
            "num_train_batches": child_model.num_train_batches,
            "preds_train": child_model.train_preds,
            "input_train": child_model.x_train,
            "label_train": child_model.y_train,
            "preds_valid": child_model.valid_preds,
            "input_valid": child_model.x_valid,
            "label_valid": child_model.y_valid
        }

        ops = {
            "child": child_ops,
            "controller": controller_ops,
            "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
            "eval_func": child_model.eval_once,
            "num_train_batches": child_model.num_train_batches,
        }

        child_ops = ops["child"]
        controller_ops = ops["controller"]
        # tf_variables_save = [var for var in tf.trainable_variables() if (var.name.startswith("child")) and 'block' not in var.name and 'channel_attn' not in var.name]
        tf_variables_save = [var for var in tf.trainable_variables() if (var.name.startswith("child"))]
        restorer = tf.train.Saver(var_list=tf_variables_save, max_to_keep=2)
        saver = tf.train.Saver(max_to_keep=200)
        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
            FLAGS.output_dir, save_steps=FLAGS.it_per_epoch, saver=saver)

        hooks = [checkpoint_saver_hook]
        if FLAGS.child_sync_replicas:
            sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
            hooks.append(sync_replicas_hook)
        if FLAGS.controller_training and FLAGS.controller_sync_replicas:
            sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(True)
            hooks.append(sync_replicas_hook)

        print("-" * 80)
        print("Starting session")
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.train.SingularMonitoredSession(
                config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
            if FLAGS.fine_tune:
                # ckpt_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                restorer.restore(sess, FLAGS.checkpoint)
            start_time = time.time()
            baseline = 0
            avg_loss = 1e9
            global_error = 0
            PSNR_save = []
            bgr_mean = np.array([103.154 / 255, 111.561 / 255, 114.356 / 255]).astype(np.float32)
            while True:
                # x_train, y_train, texture_train = sess.run([images["train"], labels["train"], texture_maps["train"]])
                x_train, y_train, texture_train = get_random_batch_texture_early(labels["train"], texture_maps["train"],
                                                                            FLAGS.batch_size,
                                                                            FLAGS.patch_size * FLAGS.child_upsample_size,
                                                                            FLAGS.child_upsample_size)
                # print(np.shape(x_train))
                # print(np.shape(y_train))
                # print(np.shape(texture_train))
                # texture_map = get_texture_map(x_train,False)
                x_train = x_train - bgr_mean
                y_train = y_train - bgr_mean
                if FLAGS.controller_training:
                    now_arcs = sess.run(controller_model.sample_arc)
                    run_ops = [
                        child_ops["loss"],
                        child_ops["lr"],
                        child_ops["grad_norm"],
                        child_ops["train_op"],
                        child_ops["input_train"],
                        child_ops["label_train"],
                        child_ops["preds_train"]
                    ]
                    loss, lr, gn, _, input_data, label_data, result_data = sess.run(run_ops, {
                        child_model.now_arc_Low: now_arcs[0], child_model.now_arc_Mid: now_arcs[1], child_model.now_arc_High: now_arcs[2],
                        child_model.x_train: x_train, child_model.y_train: y_train,
                        child_model.texture_map: texture_train})
                else:
                    run_ops = [
                        child_ops["loss"],
                        child_ops["lr"],
                        child_ops["grad_norm"],
                        child_ops["input_train"],
                        child_ops["label_train"],
                        child_ops["preds_train"]
                    ]
                    loss, lr, gn, input_data, label_data, result_data = sess.run(run_ops, {
                        child_model.x_train: x_train, child_model.y_train: y_train,
                        child_model.texture_map: texture_train})
                    if loss < avg_loss * FLAGS.loss_cut:
                        _ = sess.run(child_ops["train_op"], {child_model.x_train: x_train, child_model.y_train: y_train,
                                                             child_model.texture_map: texture_train})
                        global_error += loss
                        # print("global_loss = {}".format(global_error))
                    else:
                        print("batch loss is too big to training")
                        continue

                # layer_names = [var for var in tf.trainable_variables() if var.name.startswith("layer_2")]
                # layer_names = [var for var in tf.trainable_variables() if "layer_0" in var.name]
                # print(now_arc)

                global_step = sess.run(child_ops["global_step"])
                avg_loss = global_error / global_step
                # print("avg_loss = {}".format(avg_loss))

                result_psnr = calculate_psnr(result_data + bgr_mean, label_data + bgr_mean)

                if FLAGS.child_sync_replicas:
                    actual_step = global_step * FLAGS.num_aggregate
                else:
                    actual_step = global_step
                epoch = actual_step // FLAGS.it_per_epoch
                curr_time = time.time()
                if global_step % FLAGS.log_every == 0:
                    log_string = ""
                    log_string += "epoch={:<6d}".format(epoch)
                    log_string += "ch_step={:<6d}".format(global_step)
                    log_string += " loss={:<8.6f}".format(loss)
                    log_string += " lr={}".format(lr)
                    log_string += " |g|={:<8.4f}".format(gn)
                    log_string += " PSNR={:<7.3f}".format(
                        result_psnr)
                    log_string += " mins={:<10.2f}".format(
                        float(curr_time - start_time) / 60)

                    print(log_string)

                    # cv2.imshow("input", input_data[0] + bgr_mean)
                    # cv2.imshow("label", label_data[0] + bgr_mean)
                    # cv2.imshow("pred", result_data[0] + bgr_mean)
                    # cv2.imshow("texture_low", texture_train[0][0])
                    # cv2.imshow("texture_mid", texture_train[1][0])
                    # cv2.imshow("texture_high", texture_train[2][0])
                    # cv2.waitKey()

                if actual_step % FLAGS.it_per_epoch == 0:
                    if (FLAGS.controller_training and
                            epoch % FLAGS.controller_train_every == 0 and epoch > FLAGS.controller_train_start):
                        print("Epoch {}: Training controller".format(epoch))
                        # for ct_step in range(FLAGS.controller_train_steps *
                        #                      FLAGS.controller_num_aggregate):
                        for ct_step in range(FLAGS.controller_train_steps):
                            now_arcs = sess.run(controller_model.sample_arc)

                            run_ops = [
                                child_model.y_valid_rl,
                                child_model.valid_preds_rl
                            ]
                            total_psnr = 0
                            # num_valid_batches = 1
                            num_valid_batches = int(child_model.num_valid_batches//5)
                            for _ in range(num_valid_batches):
                                x_valid_rl, y_valid_rl, texture_valid_rl = get_random_batch_texture_early(labels["valid"],
                                                                                                     texture_maps[
                                                                                                         "valid"],
                                                                                                     FLAGS.eval_batch_size,
                                                                                                     FLAGS.eval_patch_size * FLAGS.child_upsample_size,
                                                                                                     FLAGS.child_upsample_size)
                                x_valid_rl = x_valid_rl - bgr_mean
                                y_valid_rl = y_valid_rl - bgr_mean
                                label_data, result_data = sess.run(run_ops,
                                                                   {child_model.now_arc_Low: now_arcs[0],
                                                                    child_model.now_arc_Mid: now_arcs[1],
                                                                    child_model.now_arc_High: now_arcs[2],
                                                                    child_model.x_valid_rl: x_valid_rl,
                                                                    child_model.y_valid_rl: y_valid_rl,
                                                                    child_model.texture_map: texture_valid_rl})
                                total_psnr += calculate_psnr(result_data + bgr_mean, label_data + bgr_mean)
                            val_PSNR = total_psnr / num_valid_batches

                            controller_step = sess.run(controller_ops["train_step"])

                            cb_reward = calculate_cb_penalty(now_arcs, child_model.num_cells, child_model.num_layers)

                            reward = val_PSNR - FLAGS.cb_rate * cb_reward
                            if baseline == 0 and controller_step == 0:
                                baseline = reward
                            else:
                                baseline -= (1 - controller_model.bl_dec) * (baseline - reward)
                            run_ops = [
                                controller_ops["loss"],
                                controller_ops["entropy"],
                                controller_ops["lr"],
                                controller_ops["grad_norm"],
                                controller_ops["skip_rate"],
                                controller_ops["train_op"],
                                controller_ops["reward"],
                                controller_ops["log_prob"]
                            ]

                            loss, entropy, lr, gn, skip, _, reward, log_prob = sess.run(run_ops, {
                                controller_ops["valid_PSNR"]: val_PSNR,
                                controller_model.baseline: baseline,
                                controller_model.complexity_based_reward: cb_reward})

                            controller_step = sess.run(controller_ops["train_step"])

                            if ct_step % FLAGS.controller_log_every == 0:
                                curr_time = time.time()
                                log_string = ""
                                log_string += "ctrl_step={:<6d}".format(controller_step)
                                log_string += " loss={:<7.3f}".format(loss)
                                log_string += " ent={:<5.2f}".format(entropy)
                                log_string += " lr={:<6.6f}".format(lr)
                                log_string += " |g|={:<8.4f}".format(gn)
                                log_string += " PSNR={:<6.4f}".format(val_PSNR)
                                log_string += " baseline={:<5.2f}".format(baseline)
                                log_string += " mins={:<.2f}".format(
                                    float(curr_time - start_time) / 60)
                                log_string += " reward={}".format(reward)
                                log_string += " log_prob={}".format(log_prob)
                                print(log_string)

                        print("Here are 5 architectures")
                        for _ in range(5):
                            now_arcs = sess.run(controller_model.sample_arc)
                            run_ops = [
                                child_ops["label_valid"],
                                child_ops["preds_valid"],
                            ]
                            total_psnr = 0
                            # num_valid_batches = child_model.num_valid_batches
                            num_valid_batches = 50
                            for _ in range(num_valid_batches):
                                x_valid, y_valid, texture_valid = get_random_batch_texture_early(labels["valid"],
                                                                                            texture_maps["valid"],
                                                                                            FLAGS.eval_batch_size,
                                                                                            FLAGS.eval_patch_size * FLAGS.child_upsample_size,
                                                                                            FLAGS.child_upsample_size)
                                x_valid = x_valid - bgr_mean
                                y_valid = y_valid - bgr_mean
                                label_data, result_data = sess.run(run_ops,
                                                                   {child_model.now_arc_Low: now_arcs[0],
                                                                    child_model.now_arc_Mid: now_arcs[1],
                                                                    child_model.now_arc_High: now_arcs[2],
                                                                    child_model.x_valid: x_valid,
                                                                    child_model.y_valid: y_valid,
                                                                    child_model.texture_map: texture_valid})
                                total_psnr += calculate_psnr(result_data + bgr_mean, label_data + bgr_mean)
                            val_PSNR = total_psnr / num_valid_batches
                            cb_reward = calculate_cb_penalty(now_arcs, child_model.num_cells, child_model.num_layers)

                            cb_reward = cb_reward * FLAGS.cb_rate
                            # label_data, result_data = sess.run(run_ops, {child_model.now_arc: now_arc})
                            # val_psnr = calculate_psnr(result_data, label_data)

                            print(np.reshape(now_arcs, [-1]))
                            print("val_PSNR={:<6.4f}, cb_penalty={:<6.4f}, reward={:<6.4f}".format(val_PSNR, cb_reward,
                                                                                                   val_PSNR - cb_reward))
                            print("-" * 80)
                        # print("Here are the best architecture")
                        # now_arc = sess.run(controller_inference_model.sample_arc)
                        # run_ops = [
                        #     child_ops["label_valid"],
                        #     child_ops["preds_valid"],
                        # ]
                        # total_psnr = 0
                        # num_valid_batches = 50
                        # for _ in range(num_valid_batches):
                        #     label_data, result_data = sess.run(run_ops, {child_model.now_arc: now_arc})
                        #     total_psnr += calculate_psnr(result_data, label_data)
                        # val_PSNR = total_psnr / num_valid_batches
                        # print(np.reshape(now_arc, [-1]))
                        # print("val_PSNR={:<6.4f}".format(val_PSNR))
                        # print("-" * 80)

                    if child_model.exist_fixed_arc or FLAGS.use_model is not None:
                        print("Epoch {}: Eval".format(epoch))
                        PSNRs = []
                        PSNRs_gt = []
                        for i in range(4):
                            num_examples = meta_data[test_set[i]]['total_data_num']
                            num_batches = num_examples
                            pred_img_op = child_model.test_preds
                            test_img = images[test_set[i]]
                            test_label = labels[test_set[i]]
                            total_PSNR = 0
                            total_SSIM = 0
                            total_PSNR_gt = 0
                            total_SSIM_gt = 0
                            output_dir = child_model.output_dir
                            texture_map = texture_maps[test_set[i]]

                            for batch_id in range(num_batches):
                                h_i, w_i, c_i = test_img[batch_id].shape
                                h_o, w_o, c_o = test_label[batch_id].shape
                                inp_test = test_img[batch_id] - bgr_mean
                                test_data = np.reshape(inp_test, [1, h_i, w_i, c_i])
                                texture_data = texture_map[batch_id]
                                # print("shape for pred texture_data : {}".format(np.shape(np.asarray(texture_data))))
                                # print("shape for true texture_data : {}".format(np.shape(np.asarray(texture_map[batch_id]))))
                                # data_for_texture_map = np.reshape(test_img[batch_id], [1, h_i, w_i, c_i])
                                # texture_map = get_texture_map(data_for_texture_map,True,2)

                                pred_img = sess.run(pred_img_op, feed_dict={child_model.x_test: test_data,
                                                                            child_model.texture_map: texture_data})
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

                                pred_img_u = (pred_img * 255).astype(np.uint8)
                                pred_img_u = bgr2y(pred_img_u)
                                label_img_u = (label_img * 255).astype(np.uint8)
                                label_img_u = bgr2y(label_img_u)

                                pred_img_y = bgr2y(pred_img)
                                label_img_y = bgr2y(label_img)

                                # cv2.imshow('pred',pred_img*255)
                                # cv2.imshow('label', label_img*255)
                                result_path = os.path.join(output_dir, "result_img")
                                if not os.path.isdir(result_path):
                                    print("Path {} does not exist. Creating.".format(result_path))
                                    os.makedirs(result_path)

                                # cv2.imwrite("./result_img/{}/{}".format(test_set[i], meta_data[test_set[i]]["img_names"][batch_id]), pred_img * 255)

                                # cv2.waitKey()
                                PSNR = calc_psnr(pred_img_y, label_img_y)
                                SSIM = np.mean(calculate_ssim(pred_img_u, label_img_u))
                                total_SSIM += SSIM
                                total_PSNR += PSNR
                                # print("image_{}'s PSNR = {}, SSIM = {}".format(batch_id, PSNR, SSIM))
                            total_exp = num_examples
                            psnr = total_PSNR / total_exp
                            ssim = total_SSIM / total_exp
                            PSNRs.append(psnr)
                            print(
                                "{} database, test_PSNR={:<6.4f}, test_SSIM = {:<6.4f}".format(test_set[i], psnr, ssim))
                        PSNR_save.append(PSNRs[0])
                        np.savetxt('{}/PSNR_save.txt'.format(output_dir), PSNR_save)
                        if PSNRs[2] + PSNRs[3] > perf_value:
                            perf_value = PSNRs[2] + PSNRs[3]
                            print("best PSNR on B100 and Urban100! at epoch_{}".format(epoch))
                        # if PSNRs[0] > perf_value:
                        #     perf_value = PSNRs[0]
                        #     print("best PSNR on Set5! at epoch_{}".format(epoch))                        
                if epoch >= FLAGS.num_epochs:
                    break


def test_mode():
    g = tf.Graph()
    with g.as_default():
        print("build controller...")
        controllerClass = ControllerNetwork

        controller_model = controllerClass(
            skip_target=FLAGS.controller_skip_target,
            skip_weight=FLAGS.controller_skip_weight,
            num_cells=5,
            num_layers=FLAGS.child_num_layers,
            num_branches=3,
            out_filters=FLAGS.child_out_filters,
            lstm_size=64,
            lstm_num_layers=1,
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

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(1):
                print("here are 1 architectures")
                arc_list = sess.run(controller_model.sample_arc)
                print(arc_list)


def inference():
    g = tf.Graph()
    with g.as_default():
        print("build controller...")
        controllerClass = ControllerInferenceNetwork

        controller_model = controllerClass(
            skip_target=FLAGS.controller_skip_target,
            skip_weight=FLAGS.controller_skip_weight,
            num_cells=FLAGS.child_num_cells,
            num_layers=FLAGS.child_num_layers,
            num_branches=FLAGS.child_num_branches,
            out_filters=FLAGS.child_out_filters,
            lstm_size=64,
            lstm_num_layers=1,
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

        print("read checkpoint files...")
        checkpoint_dir = FLAGS.checkpoint_dir

        with tf.Session() as sess:
            saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())

            # ckpt_state = tf.train.get_checkpoint_state("outputs/search_result")
            ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, ckpt_path)
            for i in range(1):
                print("here are 1 architectures")
                arc_list = sess.run(controller_model.sample_arc)
                print(arc_list)


def main(_):
    print("-" * 80)

    if FLAGS.inference_mode:
        if not os.path.isdir(FLAGS.checkpoint_dir):
            print("Path {} does not exist. need checkpoint dir to inference.".format(FLAGS.output_dir))
        inference()
    elif FLAGS.test_mode:
        test_mode()
    else:
        if not os.path.isdir(FLAGS.output_dir):
            print(FLAGS.output_dir)
            print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
            os.makedirs(FLAGS.output_dir)
        elif FLAGS.reset_output_dir:
            print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
            shutil.rmtree(FLAGS.output_dir)
            os.makedirs(FLAGS.output_dir)

        print("-" * 80)
        log_file = os.path.join(FLAGS.output_dir, "stdout")
        print("Logging to {}".format(log_file))
        sys.stdout = Logger(log_file)

        utils.print_user_flags()
        train()


if __name__ == "__main__":
    tf.app.run()
