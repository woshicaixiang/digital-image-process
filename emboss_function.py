# -*- coding: utf-8 -*-
"""
emboss_function.py 浮雕效果程序

@author: Ming Cheng

version of the Python packages:
numpy 1.17.4, OpenCV-Python 4.1.2.30
"""
import numpy as np
import cv2


def emboss(image, emboss_type=0):
    """
    浮雕效果函数，可自选使用八向浮雕和调和浮雕，默认八向浮雕
    :param image: 原始图像
    :param emboss_type: 浮雕类型，0表示八向浮雕，1为调和浮雕
    :return: 浮雕化后的图像
    """
    img_height, img_width, _ = image.shape
    # 将图像转换为灰度图像
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 利用OpenCV边界拓展函数拓展边界，便于卷积操作
    gray_border = cv2.copyMakeBorder(gray_img, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    emboss_img = np.zeros((img_height, img_width))
    # 八向浮雕操作
    if emboss_type == 0:
        for i in range(1, img_height + 1):
            for j in range(1, img_width + 1):
                p0 = int(gray_border[i - 1, j - 1])
                p1 = int(gray_border[i + 1, j + 1])
                new_pixel = p0 - p1 + 128
                if new_pixel > 255:
                    new_pixel = 255
                elif new_pixel < 0:
                    new_pixel = 0
                emboss_img[i - 1, j - 1] = new_pixel
    # 调和浮雕操作
    elif emboss_type == 1:
        for i in range(1, img_height + 1):
            for j in range(1, img_width + 1):
                p0 = int(gray_border[i, j])
                p1 = int(gray_border[i - 1, j - 1])
                p2 = int(gray_border[i + 1, j - 1])
                p3 = int(gray_border[i + 1, j + 1])
                p4 = int(gray_border[i - 1, j + 1])
                new_pixel = p0 * 4 - p1 - p2 - p3 - p4 + 128
                if new_pixel > 255:
                    new_pixel = 255
                elif new_pixel < 0:
                    new_pixel = 0
                emboss_img[i - 1, j - 1] = new_pixel
    else:
        return None
    emboss_img = emboss_img.astype('uint8')
    return emboss_img
