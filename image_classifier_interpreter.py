# !/usr/bin/env python3
# coding=utf-8
#
# All Rights Reserved
#
"""
explain image classifier with LIME

Authors: ChenChao (chenchao214@outlook.com)
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

from models.classifier import ImageClassifier
from path_utils import get_data_dir
from tool_box.cv2_utils import read_rgb

if __name__ == "__main__":
    img_classifier = ImageClassifier()
    explainer = lime_image.LimeImageExplainer()

    def explain_it(img_file: str) -> np.ndarray:
        label_index, label_name = img_classifier.test_image(img_file)
        img = read_rgb(img_file)
        explanation = explainer.explain_instance(img, img_classifier, top_labels=5, hide_color=0, num_samples=100)
        _, mask = explanation.get_image_and_mask(label_index, positive_only=True, num_features=5, hide_rest=True)
        img_norm = img / 255.
        img_wt_bdry = mark_boundaries(img_norm, mask)
        img_to_show = np.concatenate((img_norm, img_wt_bdry), axis=1)
        return img_to_show

    image_dir = os.path.join(get_data_dir(), "test_image")
    image_file = os.path.join(image_dir, "dog_smart.jpg")
    image_result = explain_it(image_file)
    plt.imshow(image_result)
    plt.savefig(os.path.join(image_dir, "explain_result.jpg"))
    plt.show()
