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
from torchvision import transforms
from torch.nn.functional import softmax

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

        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        self.img_size = 224

        self.index_to_label = _get_class_index()

    def __call__(self, img: np.ndarray):
        """
        Args:
            img: B x 224 x 224 x 3. rgb or bgr ?

        Returns:
            prob: np.ndarray. shape of B x 1000
        """
        return self._inference(img)

    def _inference(self, img: np.ndarray):
        img = np.transpose(img, (0, 3, 1, 2))
        img_tensor = torch.tensor(img, dtype=torch.float)
        img_tensor = self.normalizer(img_tensor)
        with torch.no_grad():
            logits = self.model(img_tensor)  # B x 1000
        prob = softmax(logits, dim=1).cpu().numpy()
        return prob

    def _get_label_name(self, index: int) -> str:
        return self.index_to_label[str(index)][1]

    def load_and_preprocess(self, img_file: str):
        """
        预处理,包含读取，resize和像素值归一化到0～1
        Args:
            img_file:
        Returns:
            normed: np.ndarray. shape 224 x 224 x 3
        """
        img = cv2.imread(img_file)
        # reshape
        img_resized = self._resize(img)

        # norm to [0, 1], transpose, to-tensor
        normed = img_resized / 255  # 224 x 224 x 3
        return img_resized, normed

    def _resize(self, img: np.ndarray):
        return cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

    def test_image(self, image_file: str = None) -> (int, str):
        if image_file is None:
            image_file = TEST_IMAGE_FILE
        assert os.path.exists(image_file), f"File not exist.{image_file}"
        _, img = self.load_and_preprocess(image_file)  # 预处理
        prob = self._inference(img[np.newaxis, ...])[0]
        index_max = np.argmax(prob)
        prob_max = prob[index_max]
        label_name = self._get_label_name(index_max)
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
