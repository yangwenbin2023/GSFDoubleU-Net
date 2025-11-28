import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utilsU import *
from train import create_dataset
from tensorflow.keras import backend as K


def read_image(x):
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (192, 256))
    image = np.clip(image - np.median(image) + 127, 0, 255)
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def read_mask(y):
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (192, 256))
    mask = mask.astype(np.float32)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def parse(y_pred):
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = y_pred[..., -1]
    y_pred = y_pred.astype(np.float32)
    y_pred = np.expand_dims(y_pred, axis=-1)
    return y_pred


def evaluate_normal(model, x_data, y_data):
    THRESHOLD = 0.5
    total = []
    for i, (x, y) in tqdm(enumerate(zip(x_data, y_data)), total=len(x_data)):
        x = read_image(x)
        y = read_mask(y)
        _, h, w, _ = x.shape
        y_pred1 = parse(model.predict(x)[0][..., 0])
        y_pred2 = parse(model.predict(x)[0][..., 0])

        line = np.ones((h, 10, 3)) * 255.0

        all_images = [
            x[0] * 255.0, line,
            mask_to_3d(y) * 255.0, line,
            mask_to_3d(y_pred1) * 255.0, line,
            mask_to_3d(y_pred2) * 255.0
        ]
        mask = np.concatenate(all_images, axis=1)

        cv2.imwrite(f"results/{i}.png", mask)


smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)  # True Positives
    fn = K.sum(y_true_f * (1 - y_pred_f))  # False Negatives
    return tp / (tp + fn + K.epsilon())


def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)  # True Positives
    fp = K.sum((1 - y_true_f) * y_pred_f)  # False Positives
    return tp / (tp + fp + K.epsilon())


def hausdorff_distance(y_true, y_pred):
    """计算 Hausdorff 距离（HD）"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return tf.reduce_max(tf.abs(y_true_f - y_pred_f))  # 简化计算方式


def mean_surface_distance(y_true, y_pred):
    """计算平均表面距离（MSD）"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return tf.reduce_mean(tf.abs(y_true_f - y_pred_f))


def load_model_weight(path):
    # 在加载模型时同时注册 iou_metric 和 dice_loss
    model = load_model(path, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef, 'iou_metric': iou_metric,
                                             'recall': recall, 'precision': precision,
                                             'hausdorff_distance': hausdorff_distance,
                                             'mean_surface_distance': mean_surface_distance})
    return model


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("results/")

    batch_size = 8

    test_path = "new_dataKvasir-SEG1/test/"
    test_x = sorted(glob(os.path.join(test_path, "image", "*.png")))
    print(tf.shape(test_x))
    test_y = sorted(glob(os.path.join(test_path, "mask", "*.png")))
    test_dataset = create_dataset(test_x, test_y)

    # test_dataset = test_dataset.batch(batch_size)

    test_steps = (len(test_x) // batch_size)

    if len(test_x) % batch_size != 0:
        test_steps += 1
    model = load_model_weight("final_model.h5")
    model.evaluate(test_dataset, steps=test_steps)
    evaluate_normal(model, test_x, test_y)
