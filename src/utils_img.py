import numpy as np
import cv2
import math
import time
from scipy import signal
from scipy import ndimage

def rotateImage(image, angle, center=None):
    if center is not None:
        image_center = center
    else:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def rgb2y(x):
    if x.dtype == np.uint8:
        x = np.float64(x)
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16
        y = np.round(y).astype(np.uint8)
    else:
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16 / 255

    return y


def bgr2y(x):
    if x.dtype == np.uint8:
        x = np.float64(x)
        y = 65.481 / 255. * x[:, :, 2] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 0] + 16
        y = np.round(y).astype(np.uint8)
    else:
        y = 65.481 / 255. * x[:, :, 2] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 0] + 16 / 255

    return y


def rgb2ycbcr(x):
    if x.dtype == np.uint8:
        x = np.float64(x)

        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16.
        y = np.round(y).astype(np.uint8)

        cb = -37.797 / 255. * x[:, :, 0] - 74.203 / 255. * x[:, :, 1] + 112. / 255. * x[:, :, 2] + 128.
        cr = 112. / 255. * x[:, :, 0] - 93.786 / 255. * x[:, :, 1] - 18.214 / 255. * x[:, :, 2] + 128.

        cb = np.round(cb).astype(np.uint8)
        cr = np.round(cr).astype(np.uint8)

    else:
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16. / 255.
        cb = -37.797 / 255. * x[:, :, 0] - 74.203 / 255. * x[:, :, 1] + 112. / 255. * x[:, :, 2] + 128. / 255.
        cr = 112. / 255. * x[:, :, 0] - 93.786 / 255. * x[:, :, 1] - 18.214 / 255. * x[:, :, 2] + 128. / 255.

    return np.stack([y, cb, cr], axis=2)


def bgr2ycbcr(x):
    if x.dtype == np.uint8:
        x = np.float64(x)

        y = 65.481 / 255. * x[:, :, 2] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 0] + 16.
        y = np.round(y).astype(np.uint8)

        cb = -37.797 / 255. * x[:, :, 2] - 74.203 / 255. * x[:, :, 1] + 112. / 255. * x[:, :, 0] + 128.
        cr = 112. / 255. * x[:, :, 2] - 93.786 / 255. * x[:, :, 1] - 18.214 / 255. * x[:, :, 0] + 128.

        cb = np.round(cb).astype(np.uint8)
        cr = np.round(cr).astype(np.uint8)

    else:
        y = 65.481 / 255. * x[:, :, 2] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 0] + 16. / 255.
        cb = -37.797 / 255. * x[:, :, 2] - 74.203 / 255. * x[:, :, 1] + 112. / 255. * x[:, :, 0] + 128. / 255.
        cr = 112. / 255. * x[:, :, 2] - 93.786 / 255. * x[:, :, 1] - 18.214 / 255. * x[:, :, 0] + 128. / 255.

    return np.stack([y, cb, cr], axis=2)


def ycbcr2rgb(x):
    T = np.asarray([[65.481, 128.553, 24.966], [-37.797, - 74.203, 112.], [112., -93.786, -18.214]])
    T = np.linalg.inv(T)

    if x.dtype == np.uint8:
        x = np.float64(x)

        x[:, :, 0] = x[:, :, 0] - 16.
        x[:, :, 1] = x[:, :, 1] - 128.
        x[:, :, 2] = x[:, :, 2] - 128.

        r = 255. * (T[0, 0] * x[:, :, 0] + T[0, 1] * x[:, :, 1] + T[0, 2] * x[:, :, 2])
        g = 255. * (T[1, 0] * x[:, :, 0] + T[1, 1] * x[:, :, 1] + T[1, 2] * x[:, :, 2])
        b = 255. * (T[2, 0] * x[:, :, 0] + T[2, 1] * x[:, :, 1] + T[2, 2] * x[:, :, 2])

        rgb = np.stack([r, g, b], axis=2)
        rgb = np.clip(rgb, 0., 255.)
        rgb = np.round(rgb).astype(np.uint8)

    else:
        x[:, :, 0] = x[:, :, 0] - 16. / 255.
        x[:, :, 1] = x[:, :, 1] - 128. / 255.
        x[:, :, 2] = x[:, :, 2] - 128. / 255.

        r = 255. * (T[0, 0] * x[:, :, 0] + T[0, 1] * x[:, :, 1] + T[0, 2] * x[:, :, 2])
        g = 255. * (T[1, 0] * x[:, :, 0] + T[1, 1] * x[:, :, 1] + T[1, 2] * x[:, :, 2])
        b = 255. * (T[2, 0] * x[:, :, 0] + T[2, 1] * x[:, :, 1] + T[2, 2] * x[:, :, 2])
        rgb = np.stack([r, g, b], axis=2)

    return rgb


def ycbcr2bgr(x):
    T = np.asarray([[65.481, 128.553, 24.966], [-37.797, - 74.203, 112.], [112., -93.786, -18.214]])
    T = np.linalg.inv(T)

    if x.dtype == np.uint8:
        x = np.float64(x)

        x[:, :, 0] = x[:, :, 0] - 16.
        x[:, :, 1] = x[:, :, 1] - 128.
        x[:, :, 2] = x[:, :, 2] - 128.

        r = 255. * (T[0, 0] * x[:, :, 0] + T[0, 1] * x[:, :, 1] + T[0, 2] * x[:, :, 2])
        g = 255. * (T[1, 0] * x[:, :, 0] + T[1, 1] * x[:, :, 1] + T[1, 2] * x[:, :, 2])
        b = 255. * (T[2, 0] * x[:, :, 0] + T[2, 1] * x[:, :, 1] + T[2, 2] * x[:, :, 2])

        bgr = np.stack([b, g, r], axis=2)
        bgr = np.clip(bgr, 0., 255.)
        bgr = np.round(bgr).astype(np.uint8)

    else:
        x[:, :, 0] = x[:, :, 0] - 16. / 255.
        x[:, :, 1] = x[:, :, 1] - 128. / 255.
        x[:, :, 2] = x[:, :, 2] - 128. / 255.

        r = 255. * (T[0, 0] * x[:, :, 0] + T[0, 1] * x[:, :, 1] + T[0, 2] * x[:, :, 2])
        g = 255. * (T[1, 0] * x[:, :, 0] + T[1, 1] * x[:, :, 1] + T[1, 2] * x[:, :, 2])
        b = 255. * (T[2, 0] * x[:, :, 0] + T[2, 1] * x[:, :, 1] + T[2, 2] * x[:, :, 2])
        bgr = np.stack([b, g, r], axis=2)

    return bgr

def calc_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    if img1.dtype == np.uint8:
        assert img2.dtype == np.uint8
        PIXEL_MAX = 255.0
    else:
        assert img2.dtype == np.float32
        PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def image_degradation(image, type):
    height, width, channel = image.shape
    result = image
    if type == 'x2':
        result = cv2.resize(
            cv2.resize(image, dsize=(int(1 / 2 * width), int(1 / 2 * height)), interpolation=cv2.INTER_CUBIC),
            dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    return result

def imresize(img, scale=None, kernel=None, antialiasing=True):
    method, kernel_width = (cubic, 4.0)

    img = img.transpose(2, 0, 1)
    in_C, in_H, in_W = img.shape
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, method, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, method, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = np.zeros([in_C, in_H + sym_len_Hs + sym_len_He, in_W])
    img_aug[:, sym_len_Hs:sym_len_Hs + in_H, :] = img

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = np.take(sym_patch, inv_idx, axis=1)
    img_aug[:, 0:0 + sym_len_Hs, :] = sym_patch_inv

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = np.take(sym_patch, inv_idx, axis=1)
    img_aug[:, sym_len_Hs + in_H:sym_len_Hs + in_H + sym_len_He, :] = sym_patch_inv

    out_1 = np.zeros([in_C, out_H, in_W])
    kernel_width = weights_H.shape[1]
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = np.matmul(img_aug[0, idx:idx + kernel_width, :].transpose(1, 0), (weights_H[i]))
        out_1[1, i, :] = np.matmul(img_aug[1, idx:idx + kernel_width, :].transpose(1, 0), (weights_H[i]))
        out_1[2, i, :] = np.matmul(img_aug[2, idx:idx + kernel_width, :].transpose(1, 0), (weights_H[i]))

    # process W dimension
    # symmetric copying
    out_1_aug = np.zeros([in_C, out_H, in_W + sym_len_Ws + sym_len_We])
    out_1_aug[:, :, sym_len_Ws:sym_len_Ws + in_W] = out_1

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = np.arange(sym_patch.shape[2] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = np.take(sym_patch, inv_idx, axis=2)
    out_1_aug[:, :, 0: 0 + sym_len_Ws] = sym_patch_inv

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = np.arange(sym_patch.shape[2] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = np.take(sym_patch, inv_idx, axis=2)
    out_1_aug[:, :, sym_len_Ws + in_W:sym_len_Ws + in_W + sym_len_We] = sym_patch_inv

    out_2 = np.zeros([in_C, out_H, out_W])
    kernel_width = weights_W.shape[1]
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = np.matmul(out_1_aug[0, :, idx:idx + kernel_width], (weights_W[i]))
        out_2[1, :, i] = np.matmul(out_1_aug[1, :, idx:idx + kernel_width], (weights_W[i]))
        out_2[2, :, i] = np.matmul(out_1_aug[2, :, idx:idx + kernel_width], (weights_W[i]))

    out_2 = out_2.transpose(1, 2, 0)
    return out_2


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = np.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = np.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = np.broadcast_to(left.reshape(out_length, 1), [out_length, P]) + np.broadcast_to(
        np.linspace(0, P - 1, P).reshape(
            1, P), [out_length, P])

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = np.broadcast_to(u.reshape(out_length, 1), [out_length, P]) - indices

    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * kernel(distance_to_center * scale)
    else:
        weights = kernel(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = np.sum(weights, 1).reshape(out_length, 1)
    weights = np.broadcast_to(weights / weights_sum, [out_length, P])

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = np.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices[:, 1:P - 2 + 1]
        weights = weights[:, 1:P - 2 + 1]
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices[:, 0:P - 2 + 1]
        weights = weights[:, 0: P - 2 + 1]
    # weights = weights.contiguous()
    # indices = indices.contiguous()
    sym_len_s = -np.min(indices) + 1
    sym_len_e = np.max(indices) - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))

def modcrop(imgs, modulo):
    sz=imgs.shape
    sz=np.asarray(sz)
    if len(sz)==2:
        sz = sz - sz % modulo
        out = imgs[0:sz[0], 0:sz[1]]
    elif len(sz)==3:
        szt = sz[0:2]
        szt = szt - szt % modulo
        out = imgs[0:szt[0], 0:szt[1],:]

    return out
    
def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))

def grad_map(x):

    return (x[:-1, :-1] - x[1:, :-1]) ** 2 + (x[:-1, :-1] - x[:-1, 1:]) ** 2


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def calculate_ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))


def calculate_psnr(images, labels):
    patch_num = images.shape[0]
    images = np.clip(images, -1e7, 1e7)
    psnr = 0
    for i in range(patch_num):
        img_data = images[i]
        label_data = labels[i]
        img_data_y = bgr2y(img_data)
        label_data_y = bgr2y(label_data)
        psnr += calc_psnr(img_data_y,label_data_y)
    psnr = psnr/patch_num
    return psnr

def get_texture_map(images, dense = False, scale_factor = 2):


    if dense:
        patch_num = images.shape[0]
        H = images.shape[1]
        W = images.shape[2]
        texture_map = []
        for i in range(patch_num):
            tmp_texture = np.zeros((H, W, 3), dtype=np.float32)
            ImgData = images[i]
            ImgData = np.pad(ImgData, ((15,15),(15,15),(0,0)), 'reflect')
            for j in range(H):
                for k in range(W):
                    ImgPatchData = ImgData[j:j+31,k:k+31,:]
                    ImgPatchData = np.round(np.clip(ImgPatchData*255, 0., 255.)).astype(np.uint8)
                    ImgPatchData_gray = cv2.cvtColor(ImgPatchData, cv2.COLOR_BGR2GRAY)

                    ImgGradData = grad_map(ImgPatchData_gray)

                    ImgGradData = ImgGradData.reshape(-1)
                    bins = [0, 6, 64, 255]
                    histGrad, bins = np.histogram(ImgGradData, bins)
                    histGrad = histGrad / ImgGradData.shape[0]
                    tmp_texture[j, k, 0] = histGrad[0]
                    tmp_texture[j, k, 1] = histGrad[1]
                    tmp_texture[j, k, 2] = histGrad[2]
            dim = (W*scale_factor,H*scale_factor)
            texture_img = cv2.resize(tmp_texture,dim, interpolation = cv2.INTER_CUBIC)
            texture_map.append(texture_img)
        texture_map = np.asarray(texture_map)
        texture_map = texture_map.reshape(1,patch_num,H*scale_factor,W*scale_factor,3)
        texture_map = np.swapaxes(texture_map,0,4)
    else:
        patch_num = images.shape[0]
        patch_size = images.shape[1]
        texture_map = np.zeros((3, patch_num, 1, 1, 1), dtype=np.float32)
        for i in range(patch_num):
            ImgData = images[i]
            ImgData = np.round(np.clip(ImgData*255, 0., 255.)).astype(np.uint8)
            ImgData_gray = cv2.cvtColor(ImgData, cv2.COLOR_BGR2GRAY)
            ImgGradData = grad_map(ImgData_gray)
            ImgGradData = ImgGradData.reshape(-1)
            bins = [0, 6, 64, 255]
            histGrad, bins = np.histogram(ImgGradData,bins)
            histGrad = histGrad/ImgGradData.shape[0]
            texture_map[0, i, :, :, :] = histGrad[0]
            texture_map[1, i, :, :, :] = histGrad[1]
            texture_map[2, i, :, :, :] = histGrad[2]

    return texture_map


def get_texture_map_color(images):

    patch_num = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    C = images.shape[3]
    texture_map = []
    texture_map_low = []
    texture_map_mid = []
    texture_map_high = []
    for i in range(patch_num):
        tmp_texture_low = np.zeros((H, W, C), dtype=np.float32)
        tmp_texture_mid = np.zeros((H, W, C), dtype=np.float32)
        tmp_texture_high = np.zeros((H, W, C), dtype=np.float32)
        ImgData = images[i]
        ImgData = np.round(np.clip(np.pad(ImgData, ((15,15),(15,15),(0,0)), 'reflect')*255, 0., 255.)).astype(np.uint8)
        GradImgData = grad_map(ImgData)
        bins = [6, 64]
        GradImgData = np.digitize(GradImgData,bins)
        for j in range(H):
            for k in range(W):
                for l in range(C):
                    ImgGradData = GradImgData[j:j+31,k:k+31,l]
                    ImgGradData = ImgGradData.reshape(-1)
                    histGrad = np.bincount(ImgGradData)
                    histGrad = histGrad / ImgGradData.shape[0]
                    if len(histGrad) == 1:
                        histGrad = np.append(histGrad,0)
                        histGrad = np.append(histGrad,0)
                    if len(histGrad) == 2:
                        histGrad = np.append(histGrad,0)
                    tmp_texture_low[j, k, l] = histGrad[0]
                    tmp_texture_mid[j, k, l] = histGrad[1]
                    tmp_texture_high[j, k, l] = histGrad[2]
        texture_map_low.append(tmp_texture_low)
        texture_map_mid.append(tmp_texture_mid)
        texture_map_high.append(tmp_texture_high)
    texture_map_low = np.asarray(texture_map_low)
    texture_map_mid = np.asarray(texture_map_mid)
    texture_map_high = np.asarray(texture_map_high)
    texture_map.append(texture_map_low)
    texture_map.append(texture_map_mid)
    texture_map.append(texture_map_high)

    return texture_map
    
def get_texture_map_color_v2(images, size = 5):

    patch_num = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    C = images.shape[3]
    texture_map = []
    texture_map_low = []
    texture_map_mid = []
    texture_map_high = []
    pad_size = int((size - 1)/2)
    for i in range(patch_num):
        tmp_texture_low = np.zeros((H, W, C), dtype=np.float32)
        tmp_texture_mid = np.zeros((H, W, C), dtype=np.float32)
        tmp_texture_high = np.zeros((H, W, C), dtype=np.float32)
        ImgData = images[i]
        ImgData = np.round(np.clip(np.pad(ImgData, ((pad_size,pad_size),(pad_size,pad_size),(0,0)), 'reflect')*255, 0., 255.)).astype(np.uint8)
        GradImgData = grad_map(ImgData)
        bins = [6, 64]
        GradImgData = np.digitize(GradImgData,bins)
        for j in range(H):
            for k in range(W):
                for l in range(C):
                    ImgGradData = GradImgData[j:j+size,k:k+size,l]
                    ImgGradData = ImgGradData.reshape(-1)
                    histGrad = np.bincount(ImgGradData)
                    histGrad = histGrad / ImgGradData.shape[0]
                    if len(histGrad) == 1:
                        histGrad = np.append(histGrad,0)
                        histGrad = np.append(histGrad,0)
                    if len(histGrad) == 2:
                        histGrad = np.append(histGrad,0)
                    tmp_texture_low[j, k, l] = histGrad[0]
                    tmp_texture_mid[j, k, l] = histGrad[1]
                    tmp_texture_high[j, k, l] = histGrad[2]
        texture_map_low.append(tmp_texture_low)
        texture_map_mid.append(tmp_texture_mid)
        texture_map_high.append(tmp_texture_high)
    texture_map_low = np.asarray(texture_map_low)
    texture_map_mid = np.asarray(texture_map_mid)
    texture_map_high = np.asarray(texture_map_high)
    texture_map.append(texture_map_low)
    texture_map.append(texture_map_mid)
    texture_map.append(texture_map_high)

    return texture_map    

