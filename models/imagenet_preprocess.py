# !/usr/bin/env python3
# coding=utf-8
#
# All Rights Reserved
#
"""
imageNet数据集，前处理相关功能函数

Authors: ChenChao (chenchao214@outlook.com)
"""
import torch
import torch.nn as nn
from torchvision import transforms


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([0.229, 0.224, 0.225])

    def forward(self, x):
        """
        Args:
            x: Batch x c x h x w
        Returns:
            same shape as x
        """
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


# read the image, resize to 224 and convert to PyTorch Tensor
preprocess = transforms.Compose([
    transforms.ToTensor(),  # 内部会自动检测数据范围，并scale到（0， 1）
    transforms.Resize((224, 224)),
])


if __name__ == "__main__":
    from models.classifier import TEST_IMAGE_FILE
    from tool_box.cv2_utils import read_rgb

    img = read_rgb(TEST_IMAGE_FILE)
    norm = Normalize()
    preprocessed = preprocess(img)
    normed = norm(preprocessed)

    # plt
    from matplotlib import pyplot as plt
    plt.imshow(preprocessed.numpy().transpose(1, 2, 0))
    plt.show()
