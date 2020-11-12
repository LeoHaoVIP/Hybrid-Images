# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from skimage import transform

np.set_printoptions(threshold=np.inf)

'''
summary：FFT计算后的高频和低频信息，先不要取实部real，否则得到的混合图像效果会很差
'''


# 加载待混合的图像（灰度图）
def load_images_gray(path1, path2):
    return cv2.imread(path1, cv2.IMREAD_GRAYSCALE), cv2.imread(path2, cv2.IMREAD_GRAYSCALE)


# 生成二维高斯核（归一化的）
def get_gaussian_filter(rows_num, cols_num, sigma):
    # rows_num和cols_num为奇数，这样核才有中心
    # rows_num * 1 gaussian kernel
    gaussian_filter_x = cv2.getGaussianKernel(rows_num, sigma)
    # cols_num * 1 gaussian kernel
    gaussian_filter_y = cv2.getGaussianKernel(cols_num, sigma)
    # rows_num * cols_num
    return gaussian_filter_x * np.transpose(gaussian_filter_y)


# 生成二维高斯核（未归一化）
def create_filter(rows_num, cols_num, sigma, high_pass=False):
    center_i = int(rows_num / 2)
    center_j = int(cols_num / 2)

    def gaussian(i, j):
        coefficient = np.exp(-1.0 * ((i - center_i) ** 2 + (j - center_j) ** 2) / (2 * sigma ** 2))
        return 1 - coefficient if high_pass else coefficient

    return np.array([[gaussian(i, j) for j in range(cols_num)] for i in range(rows_num)])


# 边界填充（使用零填充）
def pad_image_with_0(image, kernel):
    m = image.shape[0]
    n = image.shape[1]
    # 行偏差
    m_bias = int((kernel.shape[0] - 1) / 2)
    # 列偏差
    n_bias = int((kernel.shape[1] - 1) / 2)
    img_padded = np.zeros((m + m_bias * 2, n + n_bias * 2), np.uint8)
    for i in range(m):
        for j in range(n):
            img_padded[i + m_bias, j + n_bias] = image[i, j]
    return img_padded


# 自定义滤波函数（利用FFT加速卷积运算）
def my_filter_fft(image, sigma, high_pass):
    # 利用FFT快速傅里叶变换加速卷积过程
    # 高斯核(此时高斯核的大小应与图像保持一致)
    kernel = create_filter(image.shape[0], image.shape[1], sigma, high_pass=high_pass)
    shifted_dft = fftshift(fft2(image))
    filtered_dft = shifted_dft * kernel
    img_filtered = ifft2(ifftshift(filtered_dft))
    return img_filtered


# 自定义滤波函数（循环依次计算每个像素点的值）
def my_filter_normal(image, sigma, high_pass):
    kernel = get_gaussian_filter(4 * sigma + 1, 4 * sigma + 1, sigma)
    m = image.shape[0]
    n = image.shape[1]
    # 行偏差
    m_bias = int((kernel.shape[0] - 1) / 2)
    # 列偏差
    n_bias = int((kernel.shape[1] - 1) / 2)
    # 边界填充
    img_padded = pad_image_with_0(image, kernel)
    # 三层循环，对每个通道的每个像素点依次计算卷积
    for i in range(m):
        for j in range(n):
            # 当前窗口构成的矩阵
            window = img_padded[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            # 使用window * kernel相对于双重循环累加计算的速度快得多（numpy优化后）
            img_padded[i + m_bias, j + n_bias] = np.sum(window * kernel)
    # 去除边界
    img_filtered = img_padded[m_bias:m_bias + m, n_bias:n_bias + n]
    return image - img_filtered if high_pass else img_filtered


# 图像混合
# 目标：保留img1的低频信息；保留img2的高频信息
# 实现：远距离看到img2；近距离看到img1
def image_hybrid(image1, image2, sigma1, sigma2, accelerate):
    if accelerate:
        # 使用FFT加速卷积运算
        img1_low_pass = my_filter_fft(image1, sigma1, high_pass=False)
        img2_high_pass = my_filter_fft(image2, sigma2, high_pass=True)
    else:
        # 普通卷积计算（依次计算每个像素点的值）
        img1_low_pass = my_filter_normal(image1, sigma1, high_pass=False)
        img2_high_pass = my_filter_normal(image2, sigma2, high_pass=True)
    return img1_low_pass, img2_high_pass, img1_low_pass + img2_high_pass


# 获取原始图片以及滤波后的频率信息
def frequency_image(image, sigma, high_pass):
    kernel = create_filter(image.shape[0], image.shape[1], sigma, high_pass=high_pass)
    shifted_dft = fftshift(fft2(image))
    filtered_dft = shifted_dft * kernel
    return np.log(shifted_dft), np.log(filtered_dft)


if __name__ == '__main__':
    # 标记是否使用FFT加速
    use_fft_accelerate = True
    dir_path = './dataset_hybrid_images'
    pics = os.listdir(dir_path)
    pics_name_pre = ['dog', 'marilyn', 'fish', 'plane', 'motorcycle', 'lwx']
    pics_name_next = ['cat', 'einstein', 'submarine', 'brid', 'bicycle', 'tujun']
    # 分别为低通、高通滤波设定截止频率
    if use_fft_accelerate:
        # for FFT 加速
        cutoffs_img1 = [15, 10, 15, 8, 15, 25]  # 减小该值，得到更加模糊的图像
        cutoffs_img2 = [7, 25, 5, 20, 25, 5]  # 减小该值，得到更加清晰的轮廓
    else:
        # for 普通卷积
        cutoffs_img1 = [7, 10, 9, 10, 6, 10]  # 增大该值，得到更加模糊的图像
        cutoffs_img2 = [7, 1, 1, 5, 1, 10]  # 增大该值，得到更加清晰的轮廓
    # 分别为低通、高通滤波设定截止频率
    plt.ion()
    for k in range(0, len(pics) - 1, 2):
        # for k in range(2, 3):
        pair_index = int(k / 2)
        # 加载图片
        img1, img2 = load_images_gray('{0}/{1}'.format(dir_path, pics[k]),
                                      '{0}/{1}'.format(dir_path, pics[k + 1]))
        img2 = cv2.resize(img2, dsize=(img1.shape[1], img1.shape[0]))
        # 获取混合图像
        img1_low_freq, img2_high_freq, hybrid_image12 = image_hybrid(img1, img2, cutoffs_img1[pair_index],
                                                                     cutoffs_img2[pair_index],
                                                                     accelerate=use_fft_accelerate)
        # --------------------------------------------------------------------
        plt.imshow(np.real(img1_low_freq), 'gray')
        plt.savefig('./hybrid_result_gray/{0}{1}_low_freq'.format(str(pair_index * 2 + 1), str(pair_index * 2 + 2)),
                    bbox_inches='tight')
        plt.show()
        plt.imshow(np.real(img2_high_freq), 'gray')
        plt.savefig('./hybrid_result_gray/{0}{1}_high_freq'.format(str(pair_index * 2 + 1), str(pair_index * 2 + 2)),
                    bbox_inches='tight')
        plt.show()
        # 获取混合图像的金字塔表示
        img = transform.resize(np.real(hybrid_image12), (512, 512))
        rows, cols = img.shape
        pyramid = tuple(transform.pyramid_gaussian(img, downscale=2))
        composite_img = np.ones((rows, cols + int(cols / 2)), dtype=np.double)
        composite_img[:rows, :cols] = pyramid[0]
        i_row = 0
        for p in pyramid[1:]:
            n_rows, n_cols = p.shape[:2]
            composite_img[i_row:i_row + n_rows, cols:cols + n_cols] = p
            i_row += n_rows
        plt.imshow(composite_img, 'gray')
        plt.savefig('./hybrid_result_gray/{0}{1}_hybird'.format(str(pair_index * 2 + 1), str(pair_index * 2 + 2)),
                    bbox_inches='tight')
        plt.show()
        # --------------------------------------------------------------------
        # 获取图像幅度谱信息
        # img_freq, img_freq_filtered = frequency_image(img2, cutoffs_img1[pair_index], True)
        # plt.imshow(img_freq, 'gray')
        # plt.show()
        # --------------------------------------------------------------------
        # plt.suptitle('(Image Pair {0}) Hybrid Result\n\n'.format(k))
        # plt.subplot(131)
        # plt.title('Low Frequencies', y=-1)
        # plt.imshow(np.real(img1_low_freq), 'gray')
        # plt.subplot(132)
        # plt.title('High Frequencies', y=-1)
        # plt.imshow(np.real(img2_high_freq), 'gray')
        # plt.subplot(133)
        # plt.title('Hybrid Image', y=-1)
        # plt.imshow(np.real(hybrid_image12), 'gray')
        # plt.pause(0.2)
        # plt.clf()
    plt.ioff()
