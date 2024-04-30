import numpy as np
import os
import gzip
from sklearn.model_selection import train_test_split
import random
class DataLoader():
    """
    数据读取
    """
    def __init__(self,path):
        self.path = path
    def load_mnist(self,path, kind='train'):
        labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
        images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 28, 28)  # 关键点
        return images, labels

    def load_data(self):
        train_images, train_labels = self.load_mnist(self.path, kind='train')
        test_images, test_labels = self.load_mnist(self.path, kind='t10k')
        train_images = train_images.reshape(train_images.shape[0],-1)
        test_images = test_images.reshape(test_images.shape[0],-1)
        # 标准化
        train_images = (train_images-train_images.mean())/train_images.std()
        test_images = (test_images-test_images.mean())/test_images.std()
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2,
                                                                              random_state=42)
        return train_images, train_labels, val_images, val_labels, test_images, test_labels
