# -*- coding: utf-8 -*-
"""
emboss_function.py 浮雕效果程序

@author: Ming Cheng

version of the Python packages:
OpenCV-Python 4.1.2.30
"""
import cv2


def camera():
    """
    调用摄像头拍照
    :param fname: 照片的文件名
    :return: 照片文件
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    picture = None
    while True:
        # 捕获每一帧图像
        ret, frame = cap.read()
        cv2.imshow('Camera', frame)
        # 获得键盘返回值
        key_value = cv2.waitKey(1) & 0xFF
        # 按下esc键，退出摄像头
        if key_value == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
        # 按下空格键，拍摄照片并储存
        elif key_value == 32:
            # 当按下空格键后释放视频捕获器
            picture = frame
            cap.release()
            cv2.destroyAllWindows()
            break
    return picture
