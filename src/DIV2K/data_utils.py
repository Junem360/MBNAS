import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import src.utils_img as utils_img
import glob
import pickle

def threaded_input_word_pipeline(base_dir,
                                 file_patterns,
                                 num_threads=4,
                                 batch_size=32,
                                 img_size=48,
                                 label_size=96,
                                 num_epochs=None,
                                 is_train=True,
                                 file_num=None):
    queue_capacity = num_threads * batch_size * 16
    # Allow a smaller final batch if we are going for a fixed number of epochs
    final_batch = (num_epochs is not None)

    data_queue, meta_data = _get_data_queue(base_dir,
                                 file_patterns,
                                 capacity=queue_capacity,
                                 num_epochs=num_epochs,
                                 is_train=is_train,
                                 file_num=file_num)

    image_ori, gt_label = _read_DIV2K_tfrecord(data_queue, img_size, label_size)
    images, labels = tf.train.shuffle_batch([image_ori, gt_label], batch_size=batch_size, capacity=queue_capacity, num_threads=num_threads,
                                            min_after_dequeue=10, allow_smaller_final_batch=final_batch)

    return images, labels, meta_data

def threaded_input_word_pipeline_texture(base_dir,
                                 file_patterns,
                                 num_threads=4,
                                 batch_size=32,
                                 img_size=48,
                                 label_size=96,
                                 num_epochs=None,
                                 is_train=True,
                                 file_num=None):
    queue_capacity = num_threads * batch_size * 16
    # Allow a smaller final batch if we are going for a fixed number of epochs
    final_batch = (num_epochs is not None)

    data_queue = _get_data_queue(base_dir,
                                 file_patterns,
                                 capacity=queue_capacity,
                                 num_epochs=num_epochs,
                                 is_train=is_train,
                                 file_num=file_num)

    image_ori, gt_label, gt_texture_low, gt_texture_mid, gt_texture_high= _read_DIV2K_tfrecord_texture(data_queue, img_size, label_size)
    images, labels, texture_low, texture_mid, texture_high = tf.train.shuffle_batch([image_ori, gt_label, gt_texture_low, gt_texture_mid, gt_texture_high], batch_size=batch_size, capacity=queue_capacity, num_threads=num_threads,
                                            min_after_dequeue=10, allow_smaller_final_batch=final_batch)
    textures = []
    textures.append(texture_low)
    textures.append(texture_mid)
    textures.append(texture_high)
    return images, labels, textures

def _read_DIV2K_tfrecord(data_queue,img_size,label_size):
    reader = tf.TFRecordReader()  # Construct a general reader
    _, example_serialized = reader.read(data_queue)

    feature_map = {
        'train/input_LR': tf.FixedLenFeature([], tf.string),
        'train/gt_HR': tf.FixedLenFeature([], tf.string),
    }
    features = tf.parse_single_example(example_serialized, feature_map)
    image = tf.decode_raw(features['train/input_LR'], tf.float32)
    label = tf.decode_raw(features['train/gt_HR'], tf.float32)
    image = tf.reshape(image, [img_size, img_size, 3])
    label = tf.reshape(label, [label_size, label_size, 3])

    return image, label

def _read_DIV2K_tfrecord_texture(data_queue,img_size,label_size):
    reader = tf.TFRecordReader()  # Construct a general reader
    _, example_serialized = reader.read(data_queue)

    feature_map = {
        'train/input_LR': tf.FixedLenFeature([], tf.string),
        'train/gt_HR': tf.FixedLenFeature([], tf.string),
        'train/texture_Low': tf.FixedLenFeature([], tf.string),
        'train/texture_Mid': tf.FixedLenFeature([], tf.string),
        'train/texture_High': tf.FixedLenFeature([], tf.string)
    }
    features = tf.parse_single_example(example_serialized, feature_map)
    image = tf.decode_raw(features['train/input_LR'], tf.float32)
    label = tf.decode_raw(features['train/gt_HR'], tf.float32)
    texture_low = tf.decode_raw(features['train/texture_Low'], tf.float32)
    texture_mid = tf.decode_raw(features['train/texture_Mid'], tf.float32)
    texture_high = tf.decode_raw(features['train/texture_High'], tf.float32)
    image = tf.reshape(image, [img_size, img_size, 3])
    label = tf.reshape(label, [label_size, label_size, 3])
    texture_low = tf.reshape(texture_low,[label_size, label_size, 3])
    texture_mid = tf.reshape(texture_mid, [label_size, label_size, 3])
    texture_high = tf.reshape(texture_high, [label_size, label_size, 3])
    return image, label, texture_low, texture_mid, texture_high

def _get_data_queue(base_dir, file_patterns=['*.tfrecord'], capacity=2 ** 15,
                    num_epochs=None, is_train=False, file_num=None):
    """Get a data queue for a list of record files"""
    # List of lists ...
    data_files = [tf.gfile.Glob(os.path.join(base_dir, file_pattern))
                  for file_pattern in file_patterns]
    meta_data = {}
    record_data_num = []

    # flatten
    data_files = [data_file for sublist in data_files for data_file in sublist]
    # if file_num is not None and file_num < len(data_files):
    #     data_files = data_files[0:file_num]
    # for file_name in data_files:
    #     record_data_num.append(len([x for x in tf.python_io.tf_record_iterator(file_name)]))
    # meta_data["data_nums"] = record_data_num
    # meta_data["total_data_num"] = sum(record_data_num)
    # meta_data["file_num"]= len(data_files)
    # meta_data["data_files"] = data_files
    data_queue = tf.train.string_input_producer(data_files,
                                                capacity=capacity,
                                                shuffle=False,
                                                num_epochs=num_epochs)
    # return data_queue, meta_data
    return data_queue

def make_eval_batch(base_dir, type, upsample_size):
    file_patterns = ['*.png', '*.bmp', '*.jpeg']
    images, labels = [], []
    meta_data = {}
    data_files = [glob.glob(os.path.join(base_dir, file_pattern))
                  for file_pattern in file_patterns]
    data_files = [data_file for sublist in data_files for data_file in sublist]
    meta_data["total_data_num"] = len(data_files)
    meta_data["file_names"] = data_files
    for file_name in data_files:
        label_img = cv2.imread(file_name).astype(np.float32)/255
        label_img = utils_img.modcrop(label_img,upsample_size)
        degrad_img = image_degradation(label_img, type)
        images.append(degrad_img)
        labels.append(label_img)
    return images, labels, meta_data

def make_eval_batch_texture_lr(base_dir, texture_dir, type, upsample_size, texture_size):
    file_patterns = ['*.png', '*.bmp', '*.jpeg']
    texture_types = ['Low', 'Mid', 'High']
    texture_patterns = ['*.txt']
    images, labels, textures= [], [], []
    meta_data = {}
    data_files = [glob.glob(os.path.join(base_dir, file_pattern))
                  for file_pattern in file_patterns]
    data_files = [data_file for sublist in data_files for data_file in sublist]
    meta_data["total_data_num"] = len(data_files)
    meta_data["file_names"] = data_files
    for i in range(len(data_files)):
        file_name = data_files[i]

        texture_file_name_Low = "{}{}".format(file_name[file_name.rfind('/') + 1:-4], "_Low_{}.png".format(texture_size))
        tmp_name_Low = os.path.join(texture_dir, "Low", texture_file_name_Low)
        tmp_texture_Low = cv2.imread(tmp_name_Low).astype(np.float32)/255

        texture_file_name_Mid = "{}{}".format(file_name[file_name.rfind('/') + 1:-4], "_Mid_{}.png".format(texture_size))
        tmp_name_Mid = os.path.join(texture_dir, "Mid", texture_file_name_Mid)
        tmp_texture_Mid = cv2.imread(tmp_name_Mid).astype(np.float32)/255

        texture_file_name_High = "{}{}".format(file_name[file_name.rfind('/') + 1:-4], "_High_{}.png".format(texture_size))
        tmp_name_High = os.path.join(texture_dir, "High", texture_file_name_High)
        tmp_texture_High = cv2.imread(tmp_name_High).astype(np.float32)/255

        label_img = cv2.imread(file_name).astype(np.float32)/255
        label_img = utils_img.modcrop(label_img,upsample_size)

        texture_data = []
        tmp_texture_Low = np.reshape(tmp_texture_Low, [1, np.shape(tmp_texture_Low)[0], np.shape(tmp_texture_Low)[1], np.shape(tmp_texture_Low)[2]])
        tmp_texture_Mid = np.reshape(tmp_texture_Mid, [1, np.shape(tmp_texture_Mid)[0], np.shape(tmp_texture_Mid)[1],
                                                       np.shape(tmp_texture_Mid)[2]])
        tmp_texture_High = np.reshape(tmp_texture_High, [1, np.shape(tmp_texture_High)[0], np.shape(tmp_texture_High)[1],
                                                       np.shape(tmp_texture_High)[2]])
        texture_data.append(tmp_texture_Low)
        texture_data.append(tmp_texture_Mid)
        texture_data.append(tmp_texture_High)

        texture_data = np.asarray(texture_data)
        degrad_img = image_degradation(label_img, type)

        images.append(degrad_img)
        labels.append(label_img)
        textures.append(texture_data)

    return images, labels, textures, meta_data

def make_eval_batch_lr(base_dir, type, upsample_size):
    file_patterns = ['*.png', '*.bmp', '*.jpeg']
    images, labels = [], []
    meta_data = {}
    data_files = [glob.glob(os.path.join(base_dir, file_pattern))
                  for file_pattern in file_patterns]
    data_files = [data_file for sublist in data_files for data_file in sublist]
    meta_data["total_data_num"] = len(data_files)
    meta_data["file_names"] = data_files
    for i in range(len(data_files)):
        file_name = data_files[i]

        label_img = cv2.imread(file_name).astype(np.float32)/255
        label_img = utils_img.modcrop(label_img,upsample_size)

        degrad_img = image_degradation(label_img, type)

        images.append(degrad_img)
        labels.append(label_img)

    return images, labels, meta_data

def image_degradation(image, type):
    height, width, channel = image.shape
    result = image
    if type == 'x2':
        result = utils_img.imresize(result, 1/2)
        result = np.round(np.clip(result * 255., 0., 255.)) / 255
        result = utils_img.imresize(result, 2)
        result = (np.round(np.clip(result * 255., 0., 255.)) / 255).astype(np.float32)
    if type == 'x2_small':
        result = utils_img.imresize(result, 1/2)
        result = (np.round(np.clip(result * 255., 0., 255.)) / 255).astype(np.float32)
    if type == 'x3_small':
        result = utils_img.imresize(result, 1/3)
        result = (np.round(np.clip(result * 255., 0., 255.)) / 255).astype(np.float32)
    if type == 'x4_small':
        result = utils_img.imresize(result, 1/4)
        result = (np.round(np.clip(result * 255., 0., 255.)) / 255).astype(np.float32)
    return result

def get_random_batch(images, batch_size, patch_size, upscale_factor):
    i = 0
    image_batch, label_batch = [], []
    while i < batch_size:
        image_idx = int(np.floor(np.random.uniform(0,len(images))))
        h, w, c = images[image_idx].shape
        h_idx = int(np.floor(np.random.uniform(0,h-patch_size)))
        w_idx = int(np.floor(np.random.uniform(0,w-patch_size)))
        rot_idx = int(np.floor(np.random.uniform(0,4)))
        flip_idx = int(np.floor(np.random.uniform(0,4)))
        patch_label = images[image_idx][h_idx:h_idx+patch_size, w_idx:w_idx+patch_size,:]
        if utils_img.gradients(patch_label) > 51.065:
            for rot in range(rot_idx):
                patch_label = np.rot90(patch_label)
            for flip in range(flip_idx):
                patch_label = patch_label[:,::-1,:]
            patch_img = utils_img.imresize(patch_label,1/upscale_factor)
            patch_img = (np.round(np.clip(patch_img, 0., 255.)) / 255).astype(np.float32)
            patch_label = (np.round(np.clip(patch_label, 0., 255.)) / 255).astype(np.float32)
            image_batch.append(patch_img)
            label_batch.append(patch_label)
            i += 1
        else:
            continue
    image_batch = np.stack(image_batch,axis=0)
    label_batch = np.stack(label_batch, axis=0)

    return image_batch, label_batch

def get_random_batch_all(images, batch_size, patch_size, upscale_factor):
    i = 0
    image_batch, label_batch = [], []
    while i < batch_size:
        image_idx = int(np.floor(np.random.uniform(0,len(images))))
        h, w, c = images[image_idx].shape
        h_idx = int(np.floor(np.random.uniform(0,h-patch_size)))
        w_idx = int(np.floor(np.random.uniform(0,w-patch_size)))
        rot_idx = int(np.floor(np.random.uniform(0,4)))
        flip_idx = int(np.floor(np.random.uniform(0,4)))
        patch_label = images[image_idx][h_idx:h_idx+patch_size, w_idx:w_idx+patch_size,:]
        for rot in range(rot_idx):
            patch_label = np.rot90(patch_label)
        for flip in range(flip_idx):
            patch_label = patch_label[:,::-1,:]
        patch_img = utils_img.imresize(patch_label,1/upscale_factor)
        patch_img = (np.round(np.clip(patch_img, 0., 255.)) / 255).astype(np.float32)
        patch_label = (np.round(np.clip(patch_label, 0., 255.)) / 255).astype(np.float32)
        image_batch.append(patch_img)
        label_batch.append(patch_label)
        i += 1

    image_batch = np.stack(image_batch,axis=0)
    label_batch = np.stack(label_batch, axis=0)

    return image_batch, label_batch

def get_random_batch_all_texture(images, textures, batch_size, patch_size, upscale_factor):
    i = 0
    image_batch, texture_batch, label_batch = [], [], []
    texture_batch_low, texture_batch_mid, texture_batch_high = [], [], []

    while i < batch_size:
        image_idx = int(np.floor(np.random.uniform(0,len(images))))
        h, w, c = images[image_idx].shape
        h_idx = int(np.floor(np.random.uniform(0,h-patch_size)))
        w_idx = int(np.floor(np.random.uniform(0,w-patch_size)))
        rot_idx = int(np.floor(np.random.uniform(0,4)))
        flip_idx = int(np.floor(np.random.uniform(0,4)))
        patch_label = images[image_idx][h_idx:h_idx+patch_size, w_idx:w_idx+patch_size,:]

        patch_texture_low = textures[image_idx][0][h_idx:h_idx+patch_size, w_idx:w_idx+patch_size,:]
        patch_texture_low = np.reshape(patch_texture_low, [patch_size, patch_size, 3])


        patch_texture_mid = textures[image_idx][1][h_idx:h_idx + patch_size, w_idx:w_idx + patch_size, :]
        patch_texture_mid = np.reshape(patch_texture_mid, [patch_size, patch_size, 3])

        patch_texture_high = textures[image_idx][2][h_idx:h_idx + patch_size, w_idx:w_idx + patch_size, :]
        patch_texture_high = np.reshape(patch_texture_high, [patch_size, patch_size, 3])

        for rot in range(rot_idx):
            patch_label = np.rot90(patch_label)
            patch_texture_low = np.rot90(patch_texture_low)
            patch_texture_mid = np.rot90(patch_texture_mid)
            patch_texture_high = np.rot90(patch_texture_high)

        for flip in range(flip_idx):
            patch_label = patch_label[:,::-1,:]
            patch_texture_low = patch_texture_low[:,::-1,:]
            patch_texture_mid = patch_texture_mid[:, ::-1, :]
            patch_texture_high = patch_texture_high[:, ::-1, :]

        patch_img = utils_img.imresize(patch_label,1/upscale_factor)
        patch_img = (np.round(np.clip(patch_img, 0., 255.)) / 255).astype(np.float32)
        patch_label = (np.round(np.clip(patch_label, 0., 255.)) / 255).astype(np.float32)
        patch_texture_low = (np.round(np.clip(patch_texture_low, 0., 255.)) / 255).astype(np.float32)
        patch_texture_mid = (np.round(np.clip(patch_texture_mid, 0., 255.)) / 255).astype(np.float32)
        patch_texture_high = (np.round(np.clip(patch_texture_high, 0., 255.)) / 255).astype(np.float32)
        label_batch.append(patch_label)
        image_batch.append(patch_img)
        texture_batch_low.append(patch_texture_low)
        texture_batch_mid.append(patch_texture_mid)
        texture_batch_high.append(patch_texture_high)
        i += 1

    texture_batch.append(texture_batch_low)
    texture_batch.append(texture_batch_mid)
    texture_batch.append(texture_batch_high)
    label_batch = np.stack(label_batch, axis=0)
    image_batch = np.stack(image_batch, axis=0)
    texture_batch = np.asarray(texture_batch)

    return image_batch, label_batch, texture_batch

def get_random_batch_texture(images, textures, batch_size, patch_size, upscale_factor):
    i = 0
    image_batch, texture_batch, label_batch = [], [], []
    texture_batch_low, texture_batch_mid, texture_batch_high = [], [], []

    while i < batch_size:
        image_idx = int(np.floor(np.random.uniform(0,len(images))))
        h, w, c = images[image_idx].shape
        h_idx = int(np.floor(np.random.uniform(0,h-patch_size)))
        w_idx = int(np.floor(np.random.uniform(0,w-patch_size)))
        rot_idx = int(np.floor(np.random.uniform(0,4)))
        flip_idx = int(np.floor(np.random.uniform(0,4)))
        patch_label = images[image_idx][h_idx:h_idx+patch_size, w_idx:w_idx+patch_size,:]
        if utils_img.gradients(patch_label) > 51.065:
            patch_texture_low = textures[image_idx][0][h_idx:h_idx+patch_size, w_idx:w_idx+patch_size,:]
            patch_texture_low = np.reshape(patch_texture_low, [patch_size, patch_size, 3])


            patch_texture_mid = textures[image_idx][1][h_idx:h_idx + patch_size, w_idx:w_idx + patch_size, :]
            patch_texture_mid = np.reshape(patch_texture_mid, [patch_size, patch_size, 3])

            patch_texture_high = textures[image_idx][2][h_idx:h_idx + patch_size, w_idx:w_idx + patch_size, :]
            patch_texture_high = np.reshape(patch_texture_high, [patch_size, patch_size, 3])

            for rot in range(rot_idx):
                patch_label = np.rot90(patch_label)
                patch_texture_low = np.rot90(patch_texture_low)
                patch_texture_mid = np.rot90(patch_texture_mid)
                patch_texture_high = np.rot90(patch_texture_high)

            for flip in range(flip_idx):
                patch_label = patch_label[:,::-1,:]
                patch_texture_low = patch_texture_low[:,::-1,:]
                patch_texture_mid = patch_texture_mid[:, ::-1, :]
                patch_texture_high = patch_texture_high[:, ::-1, :]

            patch_img = utils_img.imresize(patch_label,1/upscale_factor)
            patch_img = (np.round(np.clip(patch_img, 0., 255.)) / 255).astype(np.float32)
            patch_label = (np.round(np.clip(patch_label, 0., 255.)) / 255).astype(np.float32)
            patch_texture_low = (np.round(np.clip(patch_texture_low, 0., 255.)) / 255).astype(np.float32)
            patch_texture_mid = (np.round(np.clip(patch_texture_mid, 0., 255.)) / 255).astype(np.float32)
            patch_texture_high = (np.round(np.clip(patch_texture_high, 0., 255.)) / 255).astype(np.float32)
            label_batch.append(patch_label)
            image_batch.append(patch_img)
            texture_batch_low.append(patch_texture_low)
            texture_batch_mid.append(patch_texture_mid)
            texture_batch_high.append(patch_texture_high)
            i += 1
        else:
            continue
    texture_batch.append(texture_batch_low)
    texture_batch.append(texture_batch_mid)
    texture_batch.append(texture_batch_high)
    label_batch = np.stack(label_batch, axis=0)
    image_batch = np.stack(image_batch, axis=0)
    texture_batch = np.asarray(texture_batch)

    return image_batch, label_batch, texture_batch

def get_random_batch_texture_early(images, textures, batch_size, patch_size, upscale_factor):
    i = 0
    image_batch, texture_batch, label_batch = [], [], []
    texture_batch_low, texture_batch_mid, texture_batch_high = [], [], []

    while i < batch_size:
        image_idx = int(np.floor(np.random.uniform(0,len(images))))
        h, w, c = images[image_idx].shape
        h_idx = int(np.floor(np.random.uniform(0,h-patch_size)))
        h_idx_tex = h_idx//upscale_factor
        h_idx = h_idx_tex*upscale_factor
        w_idx = int(np.floor(np.random.uniform(0,w-patch_size)))
        w_idx_tex = w_idx//upscale_factor
        w_idx = w_idx_tex*upscale_factor
        rot_idx = int(np.floor(np.random.uniform(0,4)))
        flip_idx = int(np.floor(np.random.uniform(0,4)))
        patch_label = images[image_idx][h_idx:h_idx+patch_size, w_idx:w_idx+patch_size,:]
        if utils_img.gradients(patch_label) > 51.065:
            patch_texture_low = textures[image_idx][0][h_idx_tex:h_idx_tex+patch_size//upscale_factor, w_idx_tex:w_idx_tex+patch_size//upscale_factor,:]
            patch_texture_low = np.reshape(patch_texture_low, [patch_size//upscale_factor, patch_size//upscale_factor, 3])


            patch_texture_mid = textures[image_idx][1][h_idx_tex:h_idx_tex+patch_size//upscale_factor, w_idx_tex:w_idx_tex+patch_size//upscale_factor,:]
            patch_texture_mid = np.reshape(patch_texture_mid, [patch_size//upscale_factor, patch_size//upscale_factor, 3])

            patch_texture_high = textures[image_idx][2][h_idx_tex:h_idx_tex+patch_size//upscale_factor, w_idx_tex:w_idx_tex+patch_size//upscale_factor,:]
            patch_texture_high = np.reshape(patch_texture_high, [patch_size//upscale_factor, patch_size//upscale_factor, 3])

            for rot in range(rot_idx):
                patch_label = np.rot90(patch_label)
                patch_texture_low = np.rot90(patch_texture_low)
                patch_texture_mid = np.rot90(patch_texture_mid)
                patch_texture_high = np.rot90(patch_texture_high)

            for flip in range(flip_idx):
                patch_label = patch_label[:,::-1,:]
                patch_texture_low = patch_texture_low[:,::-1,:]
                patch_texture_mid = patch_texture_mid[:, ::-1, :]
                patch_texture_high = patch_texture_high[:, ::-1, :]

            patch_img = utils_img.imresize(patch_label,1/upscale_factor)
            patch_img = (np.round(np.clip(patch_img, 0., 255.)) / 255).astype(np.float32)
            patch_label = (np.round(np.clip(patch_label, 0., 255.)) / 255).astype(np.float32)
            patch_texture_low = (np.round(np.clip(patch_texture_low, 0., 255.)) / 255).astype(np.float32)
            patch_texture_mid = (np.round(np.clip(patch_texture_mid, 0., 255.)) / 255).astype(np.float32)
            patch_texture_high = (np.round(np.clip(patch_texture_high, 0., 255.)) / 255).astype(np.float32)
            label_batch.append(patch_label)
            image_batch.append(patch_img)
            texture_batch_low.append(patch_texture_low)
            texture_batch_mid.append(patch_texture_mid)
            texture_batch_high.append(patch_texture_high)
            i += 1
        else:
            continue
    texture_batch.append(texture_batch_low)
    texture_batch.append(texture_batch_mid)
    texture_batch.append(texture_batch_high)
    label_batch = np.stack(label_batch, axis=0)
    image_batch = np.stack(image_batch, axis=0)
    texture_batch = np.asarray(texture_batch)

    return image_batch, label_batch, texture_batch

def get_random_batch_early(images, batch_size, patch_size, upscale_factor):
    i = 0
    image_batch, label_batch = [], []

    while i < batch_size:
        image_idx = int(np.floor(np.random.uniform(0,len(images))))
        h, w, c = images[image_idx].shape
        h_idx = int(np.floor(np.random.uniform(0,h-patch_size)))
        h_idx_tex = h_idx//upscale_factor
        h_idx = h_idx_tex*upscale_factor
        w_idx = int(np.floor(np.random.uniform(0,w-patch_size)))
        w_idx_tex = w_idx//upscale_factor
        w_idx = w_idx_tex*upscale_factor
        rot_idx = int(np.floor(np.random.uniform(0,4)))
        flip_idx = int(np.floor(np.random.uniform(0,4)))
        patch_label = images[image_idx][h_idx:h_idx+patch_size, w_idx:w_idx+patch_size,:]
        if utils_img.gradients(patch_label) > 51.065:
            for rot in range(rot_idx):
                patch_label = np.rot90(patch_label)

            for flip in range(flip_idx):
                patch_label = patch_label[:,::-1,:]

            patch_img = utils_img.imresize(patch_label,1/upscale_factor)
            patch_img = (np.round(np.clip(patch_img, 0., 255.)) / 255).astype(np.float32)
            patch_label = (np.round(np.clip(patch_label, 0., 255.)) / 255).astype(np.float32)
            label_batch.append(patch_label)
            image_batch.append(patch_img)
            i += 1
        else:
            continue
    label_batch = np.stack(label_batch, axis=0)
    image_batch = np.stack(image_batch, axis=0)

    return image_batch, label_batch

def get_random_batch_TPN(labels, textures, batch_size, patch_size, upscale_factor):
    i = 0
    texture_batch, label_batch = [], []
    texture_batch_low, texture_batch_mid, texture_batch_high = [], [], []
    label_batch_low, label_batch_mid, label_batch_high = [], [], []

    while i < batch_size:
        idx = int(np.floor(np.random.uniform(0,len(labels))))

        b, h, w, c = np.shape(textures[idx])
        h_idx = int(np.floor(np.random.uniform(0,h-patch_size)))
        w_idx = int(np.floor(np.random.uniform(0,w-patch_size)))
        rot_idx = int(np.floor(np.random.uniform(0,4)))
        flip_idx = int(np.floor(np.random.uniform(0,4)))

        patch_texture_low = textures[idx][0][h_idx:h_idx+patch_size, w_idx:w_idx+patch_size,:]
        patch_texture_low = np.reshape(patch_texture_low, [patch_size, patch_size, 3])

        patch_texture_mid = textures[idx][1][h_idx:h_idx + patch_size, w_idx:w_idx + patch_size, :]
        patch_texture_mid = np.reshape(patch_texture_mid, [patch_size, patch_size, 3])

        patch_texture_high = textures[idx][2][h_idx:h_idx + patch_size, w_idx:w_idx + patch_size, :]
        patch_texture_high = np.reshape(patch_texture_high, [patch_size, patch_size, 3])

        patch_label_low = labels[idx][0][h_idx*upscale_factor:(h_idx+patch_size)*upscale_factor, w_idx*upscale_factor:(w_idx+patch_size)*upscale_factor,:]
        patch_label_low = np.reshape(patch_label_low, [patch_size*upscale_factor, patch_size*upscale_factor, 3])

        patch_label_mid = labels[idx][1][h_idx*upscale_factor:(h_idx + patch_size)*upscale_factor, w_idx*upscale_factor:(w_idx + patch_size)*upscale_factor, :]
        patch_label_mid = np.reshape(patch_label_mid, [patch_size*upscale_factor, patch_size*upscale_factor, 3])

        patch_label_high = labels[idx][2][h_idx*upscale_factor:(h_idx + patch_size)*upscale_factor, w_idx*upscale_factor:(w_idx + patch_size)*upscale_factor, :]
        patch_label_high = np.reshape(patch_label_high, [patch_size*upscale_factor, patch_size*upscale_factor, 3])

        for rot in range(rot_idx):
            patch_label_low = np.rot90(patch_label_low)
            patch_label_mid = np.rot90(patch_label_mid)
            patch_label_high = np.rot90(patch_label_high)
            patch_texture_low = np.rot90(patch_texture_low)
            patch_texture_mid = np.rot90(patch_texture_mid)
            patch_texture_high = np.rot90(patch_texture_high)

        for flip in range(flip_idx):
            patch_label_low = patch_label_low[:,::-1,:]
            patch_label_mid = patch_label_mid[:, ::-1, :]
            patch_label_high = patch_label_high[:, ::-1, :]
            patch_texture_low = patch_texture_low[:,::-1,:]
            patch_texture_mid = patch_texture_mid[:, ::-1, :]
            patch_texture_high = patch_texture_high[:, ::-1, :]


        patch_texture_low = (np.round(np.clip(patch_texture_low, 0., 255.)) / 255).astype(np.float32)
        patch_texture_mid = (np.round(np.clip(patch_texture_mid, 0., 255.)) / 255).astype(np.float32)
        patch_texture_high = (np.round(np.clip(patch_texture_high, 0., 255.)) / 255).astype(np.float32)

        texture_batch_low.append(patch_texture_low)
        texture_batch_mid.append(patch_texture_mid)
        texture_batch_high.append(patch_texture_high)

        patch_label_low = (np.round(np.clip(patch_label_low, 0., 255.)) / 255).astype(np.float32)
        patch_label_mid = (np.round(np.clip(patch_label_mid, 0., 255.)) / 255).astype(np.float32)
        patch_label_high = (np.round(np.clip(patch_label_high, 0., 255.)) / 255).astype(np.float32)

        label_batch_low.append(patch_label_low)
        label_batch_mid.append(patch_label_mid)
        label_batch_high.append(patch_label_high)
        i += 1

    texture_batch.append(texture_batch_low)
    texture_batch.append(texture_batch_mid)
    texture_batch.append(texture_batch_high)

    label_batch.append(label_batch_low)
    label_batch.append(label_batch_mid)
    label_batch.append(label_batch_high)

    texture_batch = np.asarray(texture_batch)

    label_batch = np.asarray(label_batch)

    return label_batch, texture_batch

def get_random_batch_multiscale(images, batch_size, patch_size, upscale_factor):
    i = 0
    image_batch, label_batch = [], []
    while i < batch_size:
        image_idx = int(np.floor(np.random.uniform(0,len(images))))
        scale_idx = int(np.floor(np.random.uniform(1,4)))
        h, w, c = images[image_idx].shape
        h_idx = int(np.floor(np.random.uniform(0,h-patch_size*scale_idx)))
        w_idx = int(np.floor(np.random.uniform(0,w-patch_size*scale_idx)))
        rot_idx = int(np.floor(np.random.uniform(0,4)))
        flip_idx = int(np.floor(np.random.uniform(0,4)))
        if patch_size*scale_idx > h or patch_size*scale_idx > w: 
            continue
        else:
            patch_label = images[image_idx][h_idx:h_idx+patch_size*scale_idx, w_idx:w_idx+patch_size*scale_idx,:]
        if utils_img.gradients(patch_label) > 51.065:
            for rot in range(rot_idx):
                patch_label = np.rot90(patch_label)
            for flip in range(flip_idx):
                patch_label = patch_label[:,::-1,:]
            patch_img = utils_img.imresize(patch_label,1/(upscale_factor*scale_idx))
            patch_img = (np.round(np.clip(patch_img, 0., 255.)) / 255).astype(np.float32)
            if scale_idx is not 1:
                patch_label = utils_img.imresize(patch_label,1/scale_idx)
            patch_label = (np.round(np.clip(patch_label, 0., 255.)) / 255).astype(np.float32)
            image_batch.append(patch_img)
            label_batch.append(patch_label)
            i += 1
        else:
            continue
    image_batch = np.stack(image_batch,axis=0)
    label_batch = np.stack(label_batch, axis=0)

    return image_batch, label_batch

class DataLoader(object):
    def __init__(self,
                 batch_size = 16,
                 height = 96,
                 width = 96,
                 channel = 3,
                 scale = 2,
                 tfrecord_path = './a.tfrecord'):
        self.BATCH_SIZE = batch_size
        self.HEIGHT = height
        self.WIDTH = width
        self.CHANNEL = channel

        self.scale = scale
        self.tfrecord_path = tfrecord_path

    '''Load TFRECORD'''

    def _parse_function(self, example_proto):
        keys_to_features = {'label': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        img = parsed_features['image']
        img = tf.divide(tf.cast(tf.decode_raw(img, tf.uint8), tf.float32), 255.)
        img = tf.reshape(img, [self.HEIGHT // self.scale, self.WIDTH // self.scale, self.CHANNEL])

        label = parsed_features['label']
        label = tf.divide(tf.cast(tf.decode_raw(label, tf.uint8), tf.float32), 255.)
        label = tf.reshape(label, [self.HEIGHT, self.WIDTH, self.CHANNEL])

        return label, img

    def load_tfrecord(self, shuffle=True):
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset = dataset.map(self._parse_function)

        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()

        label_train, input_train = iterator.get_next()
        meta_data = {}
        meta_data["data_nums"] = 800000
        meta_data["total_data_num"] = 800000
        meta_data["file_num"] = 1
        meta_data["data_files"] = 'train_SR_aug_X2_80000.tfrecord'

        return input_train, label_train, meta_data

class DataLoader_float(object):
    def __init__(self,
                 batch_size = 16,
                 height = 96,
                 width = 96,
                 channel = 3,
                 scale = 2,
                 tfrecord_path = './a.tfrecord'):
        self.BATCH_SIZE = batch_size
        self.HEIGHT = height
        self.WIDTH = width
        self.CHANNEL = channel

        self.scale = scale
        self.tfrecord_path = tfrecord_path

    '''Load TFRECORD'''

    def _parse_function(self, example_proto):
        keys_to_features = {'label': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        img = parsed_features['image']
        img = tf.decode_raw(img, tf.float32)
        img = tf.reshape(img, [self.HEIGHT // self.scale, self.WIDTH // self.scale, self.CHANNEL])

        label = parsed_features['label']
        label = tf.decode_raw(label, tf.float32)
        label = tf.reshape(label, [self.HEIGHT, self.WIDTH, self.CHANNEL])

        return label, img

    def load_tfrecord(self, shuffle=True):
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset = dataset.map(self._parse_function)

        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()

        label_train, input_train = iterator.get_next()
        meta_data = {}
        meta_data["data_nums"] = 159750
        meta_data["total_data_num"] = 159750
        meta_data["file_num"] = 1
        meta_data["data_files"] = 'DIV2K_train_x4_gradual.tfrecords'

        return input_train, label_train, meta_data

def data_augmentation(img, label, batch_size):
    for i in range(batch_size):
        imgData = img[i]
        labelData = label[i]
        rot_idx = int(np.floor(np.random.uniform(0,4)))
        flip_idx = int(np.floor(np.random.uniform(0,1)))
        for rot in range(rot_idx):
            imgData = np.rot90(imgData)
            labelData = np.rot90(labelData)
        for flip in range(flip_idx):
            imgData = imgData[:, ::-1, :]
            labelData = labelData[:, ::-1, :]
        img[i] = imgData
        label[i] = labelData
    return img, label