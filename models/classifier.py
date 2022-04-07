# !/usr/bin/env python3
# coding=utf-8
#
# All Rights Reserved
#
"""
classifier inference tool based on pytorch

基于pytorch封装一个分类器用于推理

Authors: ChenChao (chenchao214@outlook.com)
"""
import os

import cv2
import numpy as np
import torch
import torchvision.models as models
from torch.nn.functional import softmax

from models.imagenet_preprocess import preprocess, Normalize
from tool_box import file_utils
from path_utils import get_data_dir

TEST_IMAGE_FILE = os.path.join(get_data_dir(), "test_image", "dog_smart.jpg")


class ImageClassifier(object):
    def __init__(self, net: str = "mobilenet_v2"):
        if net == "mobilenet_v2":
            self.model = models.mobilenet_v2(pretrained=True)
        else:
            self.model = models.efficientnet_b2(pretrained=True)

        self.model.eval()

        self.normalizer = Normalize()

        self.index_to_label = _get_class_index()

    def __call__(self, img: np.ndarray):
        """
        Args:
            img: h x w x 3. rgb or B x h x w x 3

        Returns:
            prob: np.ndarray. shape of B x 1000
        """
        if img.ndim == 4:
            img_batch = torch.stack(([preprocess(img_) for img_ in img]), dim=0)
        else:
            img_batch = preprocess(img)[None, ...]
        return self.inference(img_batch)

    def inference(self, img: torch.Tensor):
        """
        Args:
            img: B x 3 x 224 x 224
        Returns:
            prob: np.ndarray
        """
        with torch.no_grad():
            logits = self.inference_with_grad(img)
        prob = softmax(logits, dim=1)
        prob = prob.cpu().numpy()
        return prob

    def inference_with_grad(self, img: torch.Tensor):
        """
        Args:
            img: B x 3 x 224 x 224
        Returns:
        """
        img_tensor = self.normalizer(img)
        logits = self.model(img_tensor)  # B x 1000
        return logits

    def get_label_name(self, index: int) -> str:
        return self.index_to_label[str(index)][1]

    @staticmethod
    def load_and_preprocess(img_file: str):
        """
        预处理,包含读取，resize
        Args:
            img_file:
        Returns:
            img_resized : torch.Tensor. 3 x 224 x 224
        """
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = preprocess(img)
        return img_resized

    def test_image(self, image_file: str = None) -> (int, str):
        if image_file is None:
            image_file = TEST_IMAGE_FILE
        assert os.path.exists(image_file), f"File not exist.{image_file}"
        img = self.load_and_preprocess(image_file)  # read / to-tensor / resize
        print(f"img shape: {img.shape}")
        prob = self.inference(img[None, ...])[0]
        index_max = np.argmax(prob)
        prob_max = prob[index_max]
        label_name = self.get_label_name(index_max)
        print(f"Image file: {image_file}\n"
              f"Max prob  : {prob_max}\n"
              f"Index     : {index_max}\n"
              f"Label_name: {label_name}\n")
        return index_max, label_name


def _get_class_index():
    json_file = os.path.join(get_data_dir(), "imagenet_class_index.json")
    data = file_utils.read_json(json_file)
    return data


if __name__ == "__main__":
    my_classifier = ImageClassifier()
    my_classifier.test_image(TEST_IMAGE_FILE)
