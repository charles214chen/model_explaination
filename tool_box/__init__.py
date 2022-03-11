# !/usr/bin/env python3
# coding=utf-8
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
"""
FILE-DISCRIPTION.

Authors: Chen Chao (chenchao23@baidu.com)
"""
from collections import namedtuple

Point2D = namedtuple("Point2D", ["x", "y"])  # 平面点坐标
Bbox = namedtuple("Bbox", ["left_top", "w", "h"])  # bounding box. left_top是Point2D类型，表征左上角坐标
