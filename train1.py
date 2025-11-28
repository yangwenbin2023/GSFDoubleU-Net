import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import *
from glob import glob
from sklearn.model_selection import train_test_split
from model1 import build_model  # 修改此处以使用 model1.py
from utilsU import shuffling
from utils import *
from metrics import *
from tensorflow.keras.metrics import Recall

# GPU配置（可选）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置为按需增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"使用 GPU: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')


def read_image(x, target_size=(512, 384)):
    x = x.decode()
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image) + 127, 0, 255)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def read_mask(y, target_size=(512, 384)):
    y = y.decode()
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size)
    mask = mask / 255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def parse_data(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y, y  # 输出两个相同的 y

    x, y1, y2 = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32, tf.float32])

    x.set_shape([384, 512, 3])
    y1.set_shape([384, 512, 1])
    y2.set_shape([384, 512, 1])

    return x, {"segmentation": y1, "explanation": y2}  # 确保输出符合 model1.py 的结构


def create_dataset(image_paths, mask_paths, augment=False, shuffle=False, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_image_mask, num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=100, seed=SEED)

    if augment:
        def augment_func(x, y):
            if tf.random.uniform(()) > 0.5:
                x = tf.image.flip_left_right(x)
                y = tf.image.flip_left_right(y)
            if tf.random.uniform(()) > 0.5:
                x = tf.image.flip_up_down(x)
                y = tf.image.flip_up_down(y)
            k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
            x = tf.image.rot90(x, k)
            y = tf.image.rot90(y, k)
            if tf.random.uniform(()) > 0.5:
                scale = tf.random.uniform((), 0.8, 1.2)
                new_height = tf.cast(IMG_HEIGHT * scale, tf.int32)
                new_width = tf.cast(IMG_WIDTH * scale, tf.int32)
                x = tf.image.resize(x, [new_height, new_width], method='bilinear')
                y = tf.image.resize(y, [new_height, new_width], method='nearest')
                x = tf.image.resize(x, [IMG_HEIGHT, IMG_WIDTH], method='bilinear')
                y = tf.image.resize(y, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
            return x, y

        dataset = dataset.map(augment_func, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)  # 使用传递的 batch_size
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    train_path = "new_dataKvasir-SEG/train/"
    valid_path = "new_dataKvasir-SEG/valid/"

    train_x = sorted(glob(os.path.join(train_path, "image", "*.jpg")))
    train_y = sorted(glob(os.path.join(train_path, "mask", "*.jpg")))
    train_x, train_y = shuffling(train_x, train_y)

    valid_x = sorted(glob(os.path.join(valid_path, "image", "*.jpg")))
    valid_y = sorted(glob(os.path.join(valid_path, "mask", "*.jpg")))

    model_path = "Kvasirmodel.h5"
    batch_size = 2
    epochs = 300
    lr = 1e-4

    model = build_model((384, 512, 3))

    model.compile(loss={"segmentation": dice_loss, "explanation": "mse"},
                  optimizer=Adam(lr),
                  metrics={"segmentation": [dice_coef, iou, Recall(), Precision()],
                           "explanation": []})

    train_dataset = create_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = create_dataset(valid_x, valid_y, batch=batch_size)

    callbacks = [
        ModelCheckpoint(model_path),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20),
        CSVLogger("files/data1.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False)
    ]

    train_steps = (len(train_x) // batch_size) + (1 if len(train_x) % batch_size != 0 else 0)
    valid_steps = (len(valid_x) // batch_size) + (1 if len(valid_x) % batch_size != 0 else 0)

    model.fit(train_dataset,
              epochs=epochs,
              validation_data=valid_dataset,
              steps_per_epoch=train_steps,
              validation_steps=valid_steps,
              callbacks=callbacks,
              shuffle=False)
