import os
import sys

import numpy as np
import tensorflow as tf

from src.utils import count_model_params
from src.utils import get_train_ops
from src.utils_img import bgr2y
from src.utils_img import calc_psnr
import cv2

class Model(object):
    def __init__(self,
                 images,
                 labels,
                 meta_data,
                 output_dir="./outputs",
                 cutout_size=None,
                 use_model=None,
                 batch_size=32,
                 eval_batch_size=100,
                 test_batch_size=1,
                 clip_mode=None,
                 grad_bound=None,
                 it_per_epoch=1000,
                 l2_reg=1e-4,
                 lr_init=1e-4,
                 lr_dec_start=0,
                 lr_warmup_val = None,
                 lr_warmup_steps = 5,
                 lr_dec_every=100,
                 lr_dec_rate=0.1,
                 lr_dec_min = 1e-5,
                 # keep_prob=1.0,
                 optim_algo=None,
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 data_format="NHWC",
                 name="generic_model",
                 seed=None,
                 ):
        """
    Args:
      lr_dec_every: number of epochs to decay
    """
        print ("-" * 80)
        print ("Build model {}".format(name))

        self.output_dir = output_dir
        self.cutout_size = cutout_size
        self.use_model = use_model
        self.batch_size = batch_size
        self.meta_data = meta_data
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.clip_mode = clip_mode
        self.grad_bound = grad_bound
        self.l2_reg = l2_reg
        self.lr_init = lr_init
        self.lr_dec_start = lr_dec_start
        self.lr_dec_rate = lr_dec_rate
        self.lr_dec_min = lr_dec_min
        self.it_per_epoch = it_per_epoch
        # self.keep_prob = keep_prob
        self.optim_algo = optim_algo
        self.sync_replicas = sync_replicas
        self.num_aggregate = num_aggregate
        self.num_replicas = num_replicas
        self.data_format = data_format
        self.name = name
        self.seed = seed
        self.lr_warmup_val = lr_warmup_val
        self.lr_warmup_steps = lr_warmup_steps

        self.global_step = None
        self.valid_acc = None
        self.test_acc = None
        print ("Build data ops")
        with tf.device("/device:GPU:0"):
            print("get training data...")
            # training data

            self.num_train_examples = 1000000
            self.num_train_batches = 1000
            self.x_train = tf.placeholder(tf.float32, [None, None, None, 3])
            self.y_train = tf.placeholder(tf.float32, [None, None, None, 3])
            self.lr_warmup_steps = lr_warmup_steps * self.it_per_epoch
            # self.lr_dec_every = lr_dec_every * self.num_train_batches
            self.lr_dec_every = lr_dec_every * self.it_per_epoch
            self.lr_dec_start = lr_dec_start * self.it_per_epoch

            # def _pre_process(x):
            #     x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
            #     x = tf.random_crop(x, [32, 32, 3], seed=self.seed)
            #     x = tf.image.random_flip_left_right(x, seed=self.seed)
            #     if self.cutout_size is not None:
            #         mask = tf.ones([self.cutout_size, self.cutout_size], dtype=tf.int32)
            #         start = tf.random_uniform([2], minval=0, maxval=32, dtype=tf.int32)
            #         mask = tf.pad(mask, [[self.cutout_size + start[0], 32 - start[0]],
            #                              [self.cutout_size + start[1], 32 - start[1]]])
            #         mask = mask[self.cutout_size: self.cutout_size + 32,
            #                self.cutout_size: self.cutout_size + 32]
            #         mask = tf.reshape(mask, [32, 32, 1])
            #         mask = tf.tile(mask, [1, 1, 3])
            #         x = tf.where(tf.equal(mask, 0), x=x, y=tf.zeros_like(x))
            #     if self.data_format == "NCHW":
            #         x = tf.transpose(x, [2, 0, 1])
            #
            #     return x

            # self.x_train = tf.map_fn(_pre_process, x_train, back_prop=False)

            print("get validation data...")
            # valid data

            self.x_valid, self.y_valid = None, None
            if labels["valid"] is not None:
                self.num_valid_examples = self.meta_data['valid']['total_data_num']
                self.num_valid_batches = (
                        (self.num_valid_examples + self.eval_batch_size - 1)
                        // self.eval_batch_size)
                self.x_valid = tf.placeholder(tf.float32, [None, None, None, 3])
                self.y_valid = tf.placeholder(tf.float32, [None, None, None, 3])

            print("get validation data for rl...")
            # valid data for controller

            self.x_valid_rl, self.y_valid_rl = None, None
            if labels["valid_rl"] is not None:
                self.num_valid_rl_examples = self.meta_data['valid_rl']['total_data_num']
                self.num_valid_rl_batches = (
                        (self.num_valid_rl_examples + self.eval_batch_size - 1)
                        // self.eval_batch_size)
                self.x_valid_rl = tf.placeholder(tf.float32, [None, None, None, 3])
                self.y_valid_rl = tf.placeholder(tf.float32, [None, None, None, 3])

            print("get test data...")
            # test data

            self.num_test_examples = self.meta_data['test']['total_data_num']
            self.num_test_batches = self.num_test_examples
            self.x_test = tf.placeholder(tf.float32, [None, None, None, 3])
            self.y_test = tf.placeholder(tf.float32, [None, None, None, 3])

        # cache images and labels
        self.images_test = images["test"]
        self.labels_test = labels["test"]

    def eval_once(self, sess, eval_set, verbose=False):
        """Expects self.acc and self.global_step to be defined.

    Args:
      sess: tf.Session() or one of its wrap arounds.
      feed_dict: can be used to give more information to sess.run().
      eval_set: "valid" or "test"
    """

        assert self.global_step is not None
        global_step = sess.run(self.global_step)
        print ("Eval at {}".format(global_step))
        total_PSNR = 0
        total_exp = 0
        if eval_set == "valid":
            assert self.x_valid is not None
            num_batches = self.num_valid_batches
            pred_img_op = self.valid_preds
            label_img_op = self.y_valid
            for batch_id in range(num_batches):
                preds, labels = sess.run(pred_img_op, label_img_op)
                for i in range(self.eval_batch_size):
                    img_data = preds[i]
                    label_data = labels[i]
                    img_data_y = bgr2y(img_data)
                    label_data_y = bgr2y(label_data)
                    total_PSNR += calc_psnr(img_data_y, label_data_y)
                    total_exp += 1

        elif eval_set == "test":
            assert self.test_preds is not None
            num_examples = self.num_test_examples
            num_batches = self.num_test_batches
            pred_img_op = self.test_preds
            for batch_id in range(num_batches):
                h,w,c = self.images_test[batch_id].shape
                images_test = np.reshape(self.images_test[batch_id], [1,h,w,c])
                pred_img = sess.run(pred_img_op, feed_dict={self.x_test: images_test})
                pred_img = np.reshape(pred_img,[h,w,c])
                label_img = self.labels_test[batch_id]
                input_img = self.images_test[batch_id]
                pred_img_y = bgr2y(pred_img)
                label_img_y = bgr2y(label_img)
                # cv2.imshow('pred',pred_img*255)
                # cv2.imshow('label', label_img*255)
                result_path = os.path.join(self.output_dir,"result_img")
                if not os.path.isdir(result_path):
                    print("Path {} does not exist. Creating.".format(result_path))
                    os.makedirs(result_path)

                cv2.imwrite(os.path.join(self.output_dir,"result_img/{}_{}.png".format(batch_id,sess.run(self.global_step))), pred_img*255)
                cv2.imwrite(os.path.join(self.output_dir, "result_img/{}_{}_label.png".format(batch_id, sess.run(self.global_step))), label_img * 255)
                cv2.imwrite(os.path.join(self.output_dir, "result_img/{}_{}_input.png".format(batch_id, sess.run(self.global_step))), input_img * 255)

                # cv2.waitKey()
                PSNR = calc_psnr(pred_img_y,label_img_y)
                total_PSNR += PSNR
                print("image_{}'s PSNR = {}".format(batch_id,PSNR))
            total_exp = num_examples
        else:
            raise NotImplementedError("Unknown eval_set '{}'".format(eval_set))

        if verbose:
            print ("")
        print ("{}_PSNR: {:<6.3f}".format(
            eval_set, float(total_PSNR) / total_exp))

    def _build_train(self):
        print ("Build train graph")
        if self.use_model == "SRCNN":
            self.train_preds = self._model_srcnn(self.x_train, True)
        elif self.use_model == "RDN":
            self.train_preds = self._model_RDN(self.x_train, True)
        else:
            self.train_preds = self._model(self.x_train, True)
        self.loss = tf.losses.mean_squared_error(labels=self.y_train, predictions=self.train_preds)

        tf_variables = [var
                        for var in tf.trainable_variables() if var.name.startswith(self.name)]
        self.num_vars = count_model_params(tf_variables)
        print ("-" * 80)
        for var in tf_variables:
            print (var)

        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="global_step")
        self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
            self.loss,
            tf_variables,
            self.global_step,
            clip_mode=self.clip_mode,
            grad_bound=self.grad_bound,
            l2_reg=self.l2_reg,
            lr_init=self.lr_init,
            lr_dec_start=self.lr_dec_start,
            lr_warmup_steps = self.lr_warmup_steps,
            lr_warmup_val = self.lr_warmup_val,
            lr_dec_every=self.lr_dec_every,
            lr_dec_rate=self.lr_dec_rate,
            optim_algo=self.optim_algo
            # sync_replicas=self.sync_replicas,
            # num_aggregate=self.num_aggregate,
            # num_replicas=self.num_replicas
            )

    def _build_valid(self):
        if self.x_valid is not None:
            print ("-" * 80)
            print ("Build valid graph")
            if self.use_model == "SRCNN":
                self.valid_preds = self._model_srcnn(self.x_train, True)
            elif self.use_model == "RDN":
                self.valid_preds = self._model_RDN(self.x_train, True)
            else:
                self.valid_preds = self._model(self.x_train, False, reuse=True)
            self.loss = tf.losses.mean_squared_error(labels=self.y_valid, predictions=self.valid_preds)
            # self.valid_PSNR = tf.image.psnr(self.y_valid, preds, 1)
            # self.valid_PSNR = tf.reduce_sum(self.valid_PSNR)
    def _build_test(self):
        print ("-" * 80)
        print ("Build test graph")
        if self.use_model == "SRCNN":
            self.test_preds = self._model_srcnn(self.x_train, True)
        elif self.use_model == "RDN":
            self.test_preds = self._model_RDN(self.x_train, True)
        else:
            self.test_preds = self._model(self.x_train, False, reuse=True)
        # self.loss = tf.losses.mean_squared_error(labels=self.y_test, predictions=self.test_preds)
        # self.test_PSNR = tf.image.psnr(self.y_test, self.test_preds, 1)
        # self.test_PSNR = tf.reduce_sum(self.test_PSNR)
    def build_valid_rl(self, shuffle=False):
        if self.x_valid_rl is not None:
            print("-" * 80)
            print("Build valid graph")
            if self.use_model == "SRCNN":
                self.valid_preds_rl = self._model_srcnn(self.x_train, True)
            elif self.use_model == "RDN":
                self.valid_preds_rl = self._model_RDN(self.x_train, True)
            else:
                self.valid_preds_rl = self._model(self.x_valid_rl, False, reuse=True)
            self.loss = tf.losses.mean_squared_error(labels=self.y_valid_rl, predictions=self.valid_preds_rl)
            self.valid_rl_PSNR = tf.image.psnr(self.y_valid_rl, self.valid_preds_rl, 1)

    def _model(self, images, is_training, reuse=None):
        raise NotImplementedError("Abstract method")

    def _model_srcnn(self, images, is_training, reuse=None):
        raise NotImplementedError("Abstract method")

    def _model_RDN(self, images, is_training, reuse=None):
        raise NotImplementedError("Abstract method")
