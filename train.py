import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from model import build_model

# 设置随机种子以确保结果可复现
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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

# 数据路径
DATA_DIR = "new_dataKvasir-SEG1"
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train", "image")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "mask")
VALID_IMAGE_DIR = os.path.join(DATA_DIR, "valid", "image")
VALID_MASK_DIR = os.path.join(DATA_DIR, "valid", "mask")
TEST_IMAGE_DIR = os.path.join(DATA_DIR, "test", "image")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test", "mask")

# 图像参数
IMG_HEIGHT = 256  # 与数据处理脚本中的尺寸一致
IMG_WIDTH = 192
IMG_CHANNELS = 3
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE


# 定义Dice系数和Dice损失
# def dice_coef(y_true, y_pred, smooth=1):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     tf.print("y_true_f shape:", tf.shape(y_true_f))
#     tf.print("y_pred_f shape:", tf.shape(y_pred_f))
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# def dice_loss(y_true, y_pred):
#     return 1 - dice_coef(y_true, y_pred)

# # 定义IoU（Intersection over Union）
# def iou_metric(y_true, y_pred, smooth=1):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
#     return (intersection + smooth) / (union + smooth)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


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


# 加载图像和掩码路径
def get_image_mask_pairs(image_dir, mask_dir):
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                          if fname.endswith(('.png', '.jpg', '.jpeg'))])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)
                         if fname.endswith(('.png', '.jpg', '.jpeg'))])
    return image_paths, mask_paths


# 读取和预处理图像和掩码
def parse_image_mask(image_path, mask_path):
    # 确保 image_path 和 mask_path 是字符串
    image_path = tf.strings.as_string(image_path)
    mask_path = tf.strings.as_string(mask_path)

    # 读取图像
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # 确保解码为RGB
    image = tf.image.convert_image_dtype(image, tf.float32)  # 归一化到 [0,1]
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method='bilinear')

    # 读取掩码
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)  # 归一化到 [0,1]
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    mask = tf.where(mask > 0.5, 1.0, 0.0)  # 二值化

    return image, mask


# 创建TensorFlow数据集
def create_dataset(image_paths, mask_paths, augment=False, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_image_mask, num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100, seed=SEED)
    if augment:
        # 如果需要在线数据增强，请确保同步增强图像和掩码
        def augment_func(x, y):
            # 随机水平翻转
            if tf.random.uniform(()) > 0.5:
                x = tf.image.flip_left_right(x)
                y = tf.image.flip_left_right(y)

            # 随机垂直翻转
            if tf.random.uniform(()) > 0.5:
                x = tf.image.flip_up_down(x)
                y = tf.image.flip_up_down(y)

            # 随机旋转90度
            k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
            x = tf.image.rot90(x, k)
            y = tf.image.rot90(y, k)

            # 随机缩放
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
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


# 获取数据对
train_image_paths, train_mask_paths = get_image_mask_pairs(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)
valid_image_paths, valid_mask_paths = get_image_mask_pairs(VALID_IMAGE_DIR, VALID_MASK_DIR)
test_image_paths, test_mask_paths = get_image_mask_pairs(TEST_IMAGE_DIR, TEST_MASK_DIR)

# 创建数据集
# 如果已进行离线数据增强，设置 augment=False
train_dataset = create_dataset(train_image_paths, train_mask_paths, augment=False, shuffle=True)
valid_dataset = create_dataset(valid_image_paths, valid_mask_paths, augment=False, shuffle=False)
test_dataset = create_dataset(test_image_paths, test_mask_paths, augment=False, shuffle=False)

# 构建模型
model = build_model((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
model.summary()

# 编译模型
# model.compile(optimizer=Adam(learning_rate=1e-4),
#               loss='binary_crossentropy',
#               metrics=['accuracy', dice_coef, iou_metric])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=dice_loss,  # 可以使用Dice损失
              metrics=[dice_coef, iou_metric, recall, precision, hausdorff_distance, mean_surface_distance])

# 定义回调函数
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_dice_coef', verbose=1,
                             save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_dice_coef', patience=10, verbose=1, mode='max', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min')

callbacks_list = [checkpoint, earlystop, reduce_lr]

# 训练模型
EPOCHS = 300

history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=valid_dataset,
                    callbacks=callbacks_list)


# 绘制训练过程中的损失和指标
# def plot_history(history):
#     # 绘制损失
#     plt.figure(figsize=(18, 5))

#     plt.subplot(1, 3, 1)
#     plt.plot(history.history['loss'], label='训练损失')
#     plt.plot(history.history['val_loss'], label='验证损失')
#     plt.title('损失曲线')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()

#     # 绘制Dice系数
#     plt.subplot(1, 3, 2)
#     plt.plot(history.history['dice_coef'], label='训练Dice系数')
#     plt.plot(history.history['val_dice_coef'], label='验证Dice系数')
#     plt.title('Dice系数曲线')
#     plt.xlabel('Epoch')
#     plt.ylabel('Dice Coef')
#     plt.legend()

#     # 绘制IoU
#     plt.subplot(1, 3, 3)
#     plt.plot(history.history['iou_metric'], label='训练IoU')
#     plt.plot(history.history['val_iou_metric'], label='验证IoU')
#     plt.title('IoU曲线')
#     plt.xlabel('Epoch')
#     plt.ylabel('IoU')
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig('training_history.png')
#     plt.show()

def plot_history(history):
    plt.figure(figsize=(18, 6))

    metrics = ['loss', 'dice_coef', 'iou_metric', 'recall', 'precision', 'hausdorff_distance', 'mean_surface_distance']
    titles = ['Loss', 'Dice Coefficient', 'mIoU', 'Recall', 'Precision', 'Hausdorff Distance', 'Mean Surface Distance']

    for i, (metric, title) in enumerate(zip(metrics, titles), 1):
        plt.subplot(2, 4, i)
        plt.plot(history.history[metric], label=f'训练 {title}')
        plt.plot(history.history[f'val_{metric}'], label=f'验证 {title}')
        plt.title(f'{title} 曲线')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


plot_history(history)

# 加载最佳模型
model.load_weights('final_model.h5')


# 评估模型
# def evaluate_model(model, dataset, dataset_name="Dataset"):
#     loss, accuracy, dice, iou = model.evaluate(dataset)
#     print(f"{dataset_name} 评估结果:")
#     print(f"损失: {loss}")
#     print(f"准确率: {accuracy}")
#     print(f"Dice系数: {dice}")
#     print(f"IoU: {iou}")

def evaluate_model(model, dataset, dataset_name="Dataset"):
    results = model.evaluate(dataset)
    metrics = ["Loss", "Dice", "mIoU", "Recall", "Precision", "HD", "MSD"]
    print(f"\n{dataset_name} 评估结果:")
    for metric, value in zip(metrics, results):
        print(f"{metric}: {value:.4f}")


print("验证集评估结果:")
evaluate_model(model, valid_dataset, "验证集")

print("测试集评估结果:")
evaluate_model(model, test_dataset, "测试集")


# 可视化预测结果
def display_predictions(model, dataset, num=5):
    import matplotlib.pyplot as plt
    for images, masks in dataset.take(1):
        preds = model.predict(images)
        preds = tf.where(preds > 0.5, 1.0, 0.0)
        for i in range(num):
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(images[i])
            plt.title("原始图像")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(tf.squeeze(masks[i]), cmap='gray')
            plt.title("真实掩码")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(tf.squeeze(preds[i]), cmap='gray')
            plt.title("预测掩码")
            plt.axis('off')

            plt.show()


display_predictions(model, test_dataset, num=3)

# 保存最终模型
model.save('final_model.h5')
print("模型已保存为 'final_model.h5'")
