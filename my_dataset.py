from PIL import Image
from osgeo import gdal
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def read_tif(filename):
    dataset = gdal.Open(filename)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset
    return im_data


def normalize(dataset):
    for i in range(dataset.shape[2]):
        data_max = np.max(dataset[:, :, i])
        data_min = np.min(dataset[:, :, i])
        if data_max != data_min:
            dataset[:, :, i] = (dataset[:, :, i] - data_min) / (data_max - data_min)
    return dataset


class GisDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = read_tif(self.images_path[item])
        img = torch.from_numpy(normalize(img))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

