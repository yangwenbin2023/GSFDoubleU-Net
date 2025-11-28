
import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import tifffile as tif
from sklearn.model_selection import train_test_split
from utils import *
from shutil import copyfile

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightnessContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout
)

def augment_data(images, masks, save_path, augment=True):#images:待处理的图像路径列表;masks:与图像对应的标签（mask）路径列表;save_path:增强后图像和标签保存的路径;augment:是否进行数据增强
    """ Performing data augmentation. """
    crop_size = (192-32, 256-32)#中心裁剪操作,即(160,224)
    size = (256, 192)#图像和标签最终被调整到的尺寸

    for image, mask in tqdm(zip(images, masks), total=len(images)):
        image_name = image.split("/")[-1].split(".")[0]
        mask_name = mask.split("/")[-1].split(".")[0]

        x, y = read_data(image, mask)#读取指定路径的图像和标签，返回图像数据 x 和标签数据 y
        try:#获取图像的高度（h）、宽度（w）和通道数（c）
            h, w, c = x.shape
        except Exception as e:
            image = image[:-1]
            x, y = read_data(image, mask)
            h, w, c = x.shape

        if augment == True:
            ## Center Crop中心剪裁，裁剪后的尺寸为 crop_size
            aug = CenterCrop(p=1, height=crop_size[0], width=crop_size[1])
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            ## Crop从  (x_min, y_min) 到 (x_max, y_max) 区域进行裁剪
            x_min = 0
            y_min = 0
            x_max = x_min + size[0]
            y_max = y_min + size[1]

            aug = Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            ## Random Rotate 90 degree 随机的 90 度旋转
            aug = RandomRotate90(p=1)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            ## Transpose 行列互换
            aug = Transpose(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            ## ElasticTransform 进行弹性变换，模拟图像的非线性形变
            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            ## Grid Distortion 网格扭曲
            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']

            ## Optical Distortion 光学扭曲，模拟光学畸变
            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x7 = augmented['image']
            y7 = augmented['mask']

            ## Vertical Flip 垂直翻转
            aug = VerticalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x8 = augmented['image']
            y8 = augmented['mask']

            ## Horizontal Flip 水平翻转
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x9 = augmented['image']
            y9 = augmented['mask']

            ## Grayscale 将图像转换为灰度图像
            x10 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            y10 = y

            ## Grayscale Vertical Flip
            aug = VerticalFlip(p=1)
            augmented = aug(image=x10, mask=y10)
            x11 = augmented['image']
            y11 = augmented['mask']

            ## Grayscale Horizontal Flip
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x10, mask=y10)
            x12 = augmented['image']
            y12 = augmented['mask']

            ## Grayscale Center Crop 灰度图像的中心裁剪
            aug = CenterCrop(p=1, height=crop_size[0], width=crop_size[1])
            augmented = aug(image=x10, mask=y10)
            x13 = augmented['image']
            y13 = augmented['mask']

            ## 增强图像对比度
            aug = CLAHE(p=1)
            augmented = aug(image=x, mask=y)
            x14 = augmented['image']
            y14 = augmented['mask']

            ## 色相饱和度值调整
            aug = RandomGamma(p=1)
            augmented = aug(image=x, mask=y)
            x15 = augmented['image']
            y15 = augmented['mask']

            ## RGB通道进行随机偏移
            aug = HueSaturationValue(p=1)
            augmented = aug(image=x, mask=y)
            x16 = augmented['image']
            y16 = augmented['mask']

            ## 亮度对比度随机调整
            aug = RGBShift(p=1)
            augmented = aug(image=x, mask=y)
            x17 = augmented['image']
            y17 = augmented['mask']

            ## 随机裁剪
            aug = RandomBrightnessContrast(p=1)
            augmented = aug(image=x, mask=y)
            x18 = augmented['image']
            y18 = augmented['mask']

            ## 运动模糊、中值模糊、高斯模糊
            aug = RandomSizedCrop(min_max_height=(200, 300),  # 裁剪高度的范围
                                  p=1,  # 变换的应用概率
                                  size=(256, 192))  # 输出图像的大小
            augmented = aug(image=x, mask=y)
            x19 = augmented['image']
            y19 = augmented['mask']


            aug = MotionBlur(p=1, blur_limit=7)
            augmented = aug(image=x, mask=y)
            x20 = augmented['image']
            y20 = augmented['mask']

            aug = MedianBlur(p=1, blur_limit=(3,9))
            augmented = aug(image=x, mask=y)
            x21 = augmented['image']
            y21 = augmented['mask']

            aug = GaussianBlur(p=1, blur_limit=(3,9))
            augmented = aug(image=x, mask=y)
            x22 = augmented['image']
            y22 = augmented['mask']

            ## 高斯噪声
            aug = GaussNoise(p=1)
            augmented = aug(image=x, mask=y)
            x23 = augmented['image']
            y23 = augmented['mask']

            ## 通道洗牌
            aug = ChannelShuffle(p=1)
            augmented = aug(image=x, mask=y)
            x24 = augmented['image']
            y24 = augmented['mask']

            ## 粗糙丢弃
            aug = CoarseDropout(p=1, max_holes=8, max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x25 = augmented['image']
            y25 = augmented['mask']

            images = [
                x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
                x21, x22, x23, x24, x25
            ]
            masks  = [
                y, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10,
                y11, y12, y13, y14, y15, y16, y17, y18, y19, y20,
                y21, y22, y23, y24, y25
            ]

        else:
            images = [x]
            masks  = [y]

        idx = 0
        for i, m in zip(images, masks):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{image_name}_{idx}.jpg"
            tmp_mask_name  = f"{mask_name}_{idx}.jpg"

            image_path = os.path.join(save_path, "image/", tmp_image_name)
            mask_path  = os.path.join(save_path, "mask/", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1

def load_data(path, split=0.1):
    """ Load all the data and then split them into train and valid dataset. """
    img_path = glob(os.path.join(path, "images0/*"))#使其匹配指定目录下 images 文件夹中的所有文件（图像）
    msk_path = glob(os.path.join(path, "images1/*"))#匹配的是 masks 文件夹中的所有文件（标签）

    #确保图像和对应的标签按相同的顺序排列
    img_path.sort()
    msk_path.sort()

    len_ids = len(img_path)#获取图像列表的长度，即数据集中图像的数量 80%训练，10%验证，10%测试
    print(len_ids)
    train_size = int((80/100)*len_ids)
    valid_size = int((10/100)*len_ids)		## Here 10 is the percent of images used for validation
    test_size = int((10/100)*len_ids)		## Here 10 is the percent of images used for testing

    train_x, test_x = train_test_split(img_path, test_size=test_size, random_state=42) #函数将图像和标签数据划分为训练集和测试集。测试集的大小由 test_size 参数决定。
    train_y, test_y = train_test_split(msk_path, test_size=test_size, random_state=42) #random_state=42 保证每次运行代码时，分割的结果是相同的，即保证数据划分的可重复性。

    train_x, valid_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

# ## Skin lesion segmentation
# def get_skin_lesion_data(path, split=0.1):
#     train_x = glob(os.path.join(path, "trainx/*"))
#     train_y = glob(os.path.join(path, "trainy/*"))
#
#     valid_x = glob(os.path.join(path, "validationx/*"))
#     valid_y = glob(os.path.join(path, "validationy/*"))
#
#     test_x = glob(os.path.join(path, "testx/*"))
#     test_y = glob(os.path.join(path, "testy/*"))
#
#     return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
# 目录创建函数
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 读取图像和标签数据的函数
def read_data(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask

# def main():
#     np.random.seed(42)
#     path = "E:/machinelearning/kvasir-seg/Kvasir-SEG/"
#     #path = "data/skin-lesion-segmentation/"
#     (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path, split=0.1)
#
#     create_dir("new_dataKvasir-SEG/train/image/")
#     create_dir("new_dataKvasir-SEG/train/mask/")
#     create_dir("new_dataKvasir-SEG/valid/image/")
#     create_dir("new_dataKvasir-SEG/valid/mask/")
#     create_dir("new_dataKvasir-SEG/test/image/")
#     create_dir("new_dataKvasir-SEG/test/mask/")
#
#     augment_data(train_x, train_y, "new_dataKvasir-SEG/train/", augment=True)
#     augment_data(valid_x, valid_y, "new_dataKvasir-SEG/valid/", augment=False)
#     augment_data(test_x, test_y, "new_dataKvasir-SEG/test/", augment=False)

def save_raw_data(images, masks, save_path):
    """ Save original train and test images and masks. """
    create_dir(os.path.join(save_path, "image"))
    create_dir(os.path.join(save_path, "mask"))

    for image, mask in zip(images, masks):
        image_name = os.path.splitext(os.path.basename(image))[0]
        mask_name = os.path.splitext(os.path.basename(mask))[0]

        tmp_image_name = f"{image_name}.jpg"
        tmp_mask_name = f"{mask_name}.jpg"

        image_path = os.path.join(save_path, "image", tmp_image_name)
        mask_path = os.path.join(save_path, "mask", tmp_mask_name)

        # Copy original images and masks to respective folders
        copyfile(image, image_path)
        copyfile(mask, mask_path)


def main():
    np.random.seed(42)
#     path = "E:/machinelearning/kvasir-seg/Kvasir-SEG/"
#     (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path, split=0.1)

#     create_dir("new_dataKvasir-SEG/train/image/")
#     create_dir("new_dataKvasir-SEG/train/mask/")
#     create_dir("new_dataKvasir-SEG/valid/image/")
#     create_dir("new_dataKvasir-SEG/valid/mask/")
#     create_dir("new_dataKvasir-SEG/test/image/")
#     create_dir("new_dataKvasir-SEG/test/mask/")

#     # Save raw data
#     save_raw_data(train_x, train_y, "new_dataKvasir-SEG/train/")
#     save_raw_data(valid_x, valid_y, "new_dataKvasir-SEG/valid/")
#     save_raw_data(test_x, test_y, "new_dataKvasir-SEG/test/")

    # Perform augmentation (optional)
    augment_data(train_x, train_y, "new_dataKvasir-SEG/train/", augment=True)


if __name__ == "__main__":
    main()

