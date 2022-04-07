# !/usr/bin/env python3
# coding=utf-8
#
# All Rights Reserved
#
"""
opencv-based tool.
Authors: ChenChao (chenchao214@outlook.com)
"""
import cv2
import numpy as np


def resize(img: np.ndarray, h: int, w: int):
    return cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_CUBIC)


def read_rgb(img_file: str):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
