"""
Code adapted from
https://github.com/4uiiurz1/pytorch-auto-augment/blob/master/auto_augment.py
"""

import numpy as np
import random
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageOps


class RandomAugment:
    """
    Random aggressive data augmentation transformer.
    """
    def __init__(self, N=4, M=6):
        """
        :param N: int, [1, #ops]. max number of operations
        :param M: int, [0, 9]. max magnitude of operations
        """
        self.operations = {
            'ShearX': lambda img, magnitude: self.shear_x(img, magnitude),
            'ShearY': lambda img, magnitude: self.shear_y(img, magnitude),
            'TranslateX': lambda img, magnitude: self.translate_x(img, magnitude),
            'TranslateY': lambda img, magnitude: self.translate_y(img, magnitude),
            'Rotate': lambda img, magnitude: self.rotate(img, magnitude),
            'Mirror': lambda img, magnitude: self.mirror(img, magnitude),
            'Identity': lambda img, magnitude: self.identity(img, magnitude),

            'AutoContrast': lambda img, magnitude: self.auto_contrast(img, magnitude),
            'Equalize': lambda img, magnitude: self.equalize(img, magnitude),
            'Solarize': lambda img, magnitude: self.solarize(img, magnitude),
            'Posterize': lambda img, magnitude: self.posterize(img, magnitude),
            'Contrast': lambda img, magnitude: self.contrast(img, magnitude),
            'Color': lambda img, magnitude: self.color(img, magnitude),
            'Brightness': lambda img, magnitude: self.brightness(img, magnitude),
            'Sharpness': lambda img, magnitude: self.sharpness(img, magnitude),
            'Invert': lambda img, magnitude: self.invert(img, magnitude),
            # 'Gray': lambda img, magnitude: self.gray(img, magnitude)
        }

        self.N = np.clip(N, a_min=1, a_max=len(self.operations))
        self.M = np.clip(M, a_min=0, a_max=9)

    def transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = offset_matrix @ matrix @ reset_matrix
        return transform_matrix

    def shear_x(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 0.3, 11)
        transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def shear_y(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 0.3, 11)
        transform_matrix = np.array([[1, 0, 0],
                                     [random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 1, 0],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def translate_x(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 150/331, 11)
        transform_matrix = np.array([[1, 0, 0],
                                     [0, 1, img.size[1]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def translate_y(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 150/331, 11)
        transform_matrix = np.array([[1, 0, img.size[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def rotate(self, img, magnitude):
        img = img.transpose(Image.TRANSPOSE)
        magnitudes = np.random.choice([-1.0, 1.0]) * np.linspace(0, 30, 11)
        theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0],
                                     [0, 0, 1]])
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, img.size[0], img.size[1])
        img = img.transform(img.size, Image.AFFINE, transform_matrix.flatten()[:6], Image.BICUBIC)
        img = img.transpose(Image.TRANSPOSE)
        return img

    def auto_contrast(self, img, magnitude):
        img = ImageOps.autocontrast(img)
        return img

    def invert(self, img, magnitude):
        img = ImageOps.invert(img)
        return img

    def equalize(self, img, magnitude):
        img = ImageOps.equalize(img)
        return img

    def mirror(self, img, magnitude):
        img = ImageOps.mirror(img)
        return img

    def solarize(self, img, magnitude):
        magnitudes = np.linspace(0, 256, 11)
        img = ImageOps.solarize(img, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def posterize(self, img, magnitude):
        magnitudes = np.linspace(4, 8, 11)
        img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))))
        return img

    def contrast(self, img, magnitude):
        magnitudes = np.linspace(0.1, 1.9, 11)
        img = ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def color(self, img, magnitude):
        magnitudes = np.linspace(0.1, 1.9, 11)
        img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def brightness(self, img, magnitude):
        magnitudes = np.linspace(0.1, 1.9, 11)
        img = ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def sharpness(self, img, magnitude):
        magnitudes = np.linspace(0.1, 1.9, 11)
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
        return img

    def gray(self, img, magnitude):
        return ImageOps.grayscale(img).convert("RGB")

    def identity(self, img, magnitude):
        return img

    def __call__(self, img):
        ops = np.random.choice(list(self.operations.keys()), self.N)
        for op in ops:
            mag = random.randint(0, self.M)
            img = self.operations[op](img, mag)

        return img
