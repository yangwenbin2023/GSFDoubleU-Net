import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation, Multiply,
                                     GlobalAveragePooling2D, Reshape, Dense, Concatenate, UpSampling2D,
                                     MaxPool2D, GlobalMaxPooling2D, Add, AveragePooling2D, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import DepthwiseConv2D

from SqueezeUNet import SqueezeUNet


# -------------------------------
# 1. 基本模块
# -------------------------------
def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = int(init.shape[channel_axis])
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x


def conv_block(inputs, filters):
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = squeeze_excite_block(x)
    return x


# -------------------------------
# 2. 多尺度注意力模块（第一分支）
# -------------------------------
def multi_scale_attention_block(inputs):
    """通过不同尺度的卷积分支获得注意力图，然后对输入进行加权"""
    filters = int(inputs.shape[-1])

    branch1 = Conv2D(filters, (1, 1), padding="same", kernel_initializer='he_normal')(inputs)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)

    branch2 = Conv2D(filters, (3, 3), padding="same", dilation_rate=2, kernel_initializer='he_normal')(inputs)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)

    branch3 = Conv2D(filters, (3, 3), padding="same", dilation_rate=4, kernel_initializer='he_normal')(inputs)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)

    concat = Concatenate()([branch1, branch2, branch3])
    attention = Conv2D(filters, (1, 1), padding="same", activation="sigmoid", kernel_initializer='he_normal')(concat)
    out = Multiply()([inputs, attention])
    return out


# -------------------------------
# 3. 第一分支：编码器1（利用预训练的 VGG19）
# -------------------------------
def encoder1(inputs):
    skip_connections = []
    model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    # 选取部分中间层作为跳跃连接（可根据需求调整）
    names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
    for name in names:
        skip_connections.append(model.get_layer(name).output)
    output = model.get_layer("block5_conv4").output
    return output, skip_connections


# -------------------------------
# 4. 第一分支：解码器1（增加高分辨率特征重建模块）
# -------------------------------
def decoder1(inputs, skip_connections):
    num_filters = [256, 128, 64, 32]
    skip_connections = skip_connections[::-1]  # 翻转跳跃连接顺序
    x = inputs
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_connections[i]])
        # 高分辨率特征重建模块
        x = Conv2D(f, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(f, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # 后接卷积块
        x = conv_block(x, f)
    return x


# -------------------------------
# 5. 第二分支：编码器2
# -------------------------------
def encoder2(inputs):
    num_filters = [32, 64, 128, 256]
    skip_connections = []
    x = inputs
    for i, f in enumerate(num_filters):
        x = conv_block(x, f)
        skip_connections.append(x)
        x = MaxPool2D((2, 2))(x)
    return x, skip_connections


# -------------------------------
# 6. 多实例学习与自注意力模块（第二分支）
# -------------------------------
def multi_instance_self_attention_block(inputs):
    """先进行自注意力计算，再融合全局实例特征"""
    filters = int(inputs.shape[-1])
    # 将空间维度展开
    flat = Reshape((-1, filters))(inputs)  # shape: (batch, H*W, filters)

    # 自注意力部分
    query = Dense(filters // 8, use_bias=False)(flat)
    key = Dense(filters // 8, use_bias=False)(flat)
    value = Dense(filters, use_bias=False)(flat)

    attention_scores = tf.matmul(query, key, transpose_b=True)  # (batch, H*W, H*W)
    attention_scores = Activation('softmax')(attention_scores)

    attn_out = tf.matmul(attention_scores, value)  # (batch, H*W, filters)
    # 将特征恢复为原来的空间尺寸
    # 注意：这里假设输入的空间尺寸在编译时是已知的
    h = inputs.shape[1]
    w = inputs.shape[2]
    attn_out = Reshape((h, w, filters))(attn_out)

    # 多实例学习分支：全局最大池化提取实例级特征
    instance_features = GlobalMaxPooling2D()(inputs)
    instance_features = Dense(filters, activation='relu')(instance_features)
    instance_features = Dense(filters, activation='sigmoid')(instance_features)
    # 将 instance_features broadcast 到空间维度
    instance_features = Reshape((1, 1, filters))(instance_features)
    instance_features = tf.tile(instance_features, [1, h, w, 1])

    out = Add()([attn_out, Multiply()([inputs, instance_features])])
    return out


# -------------------------------
# 7. 第二分支：解码器2（增加高分辨率特征重建模块）
# -------------------------------
def decoder2(inputs, skip_1, skip_2):
    num_filters = [256, 128, 64, 32]
    # 对跳跃连接进行反转，保证空间尺寸匹配
    skip_1 = skip_1[::-1]
    skip_2 = skip_2[::-1]
    x = inputs

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_1[i], skip_2[i]])
        # 高分辨率特征重建模块
        x = Conv2D(f, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(f, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # 后接卷积块
        x = conv_block(x, f)
    return x


# -------------------------------
# 8. 输出块
# -------------------------------
def output_block(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    return x


def lightweight_aspp(x, filters):
    """轻量级 ASPP（L-ASPP），使用较小的空洞率和深度可分离卷积"""

    # 使用深度可分离卷积替代普通 1x1 卷积
    y1 = Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)

    # 3x3 空洞卷积（dilation=1）
    y2 = DepthwiseConv2D((3, 3), dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    # 3x3 空洞卷积（dilation=3）
    y3 = DepthwiseConv2D((3, 3), dilation_rate=3, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    # 3x3 空洞卷积（dilation=5）
    y4 = DepthwiseConv2D((3, 3), dilation_rate=5, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    # 轻量级通道注意力（ECA）
    y = Concatenate()([y1, y2, y3, y4])
    y = Conv2D(filters, (1, 1), padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y


def lightweight_cnn_encoder(inputs):
    """轻量级 CNN 替代传统的第二分支 CNN 编码器"""
    num_filters = [48, 96, 192, 384]  # 确保 filters 是 3 的倍数
    skip_connections = []
    x = inputs

    for i, f in enumerate(num_filters):
        # 深度可分离卷积（减少计算量）
        x = Conv2D(f, (3, 3), padding="same", kernel_initializer='he_normal', use_bias=False, groups=3)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # 空洞卷积增强感受野
        x = Conv2D(f, (3, 3), padding="same", dilation_rate=2, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        skip_connections.append(x)
        x = MaxPool2D((2, 2))(x)  # 下采样

    return x, skip_connections


# -------------------------------
# 10. 图卷积层，用于解释性分支
# -------------------------------
# class GraphConvLayer(tf.keras.layers.Layer):
#     def __init__(self, output_dim, **kwargs):
#         super(GraphConvLayer, self).__init__(**kwargs)
#         self.output_dim = output_dim

#     def build(self, input_shape):
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(input_shape[-1], self.output_dim),
#                                       initializer='glorot_uniform',
#                                       trainable=True)
#         super(GraphConvLayer, self).build(input_shape)

#     def call(self, inputs):
#         # inputs: (batch, nodes, features)
#         # 构造相似度矩阵（简单使用内积），并归一化
#         sim = tf.matmul(inputs, tf.stop_gradient(inputs), transpose_b=True)  # (batch, nodes, nodes)
#         A = tf.nn.softmax(sim, axis=-1)
#         h = tf.matmul(A, inputs)  # 消息传递
#         output = tf.matmul(h, self.kernel)
#         return output

class GraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(GraphConvLayer, self).build(input_shape)

    def call(self, inputs):
        # 计算相似度矩阵，并应用softmax
        sim = tf.matmul(inputs, inputs, transpose_b=True)  # (batch, nodes, nodes)
        A = tf.nn.softmax(sim, axis=-1)
        # 消息传递与变换
        h = tf.matmul(A, inputs)
        return tf.matmul(h, self.kernel)


def graph_conv_explain_block(inputs):
    # 增加下采样力度：两次2x2平均池化，总步长为4
    x = AveragePooling2D(pool_size=(2, 2))(inputs)
    x = AveragePooling2D(pool_size=(2, 2))(x)  # 新增的池化层

    h = x.shape[1]
    w = x.shape[2]
    filters = int(x.shape[-1])
    x_flat = Reshape((-1, filters))(x)  # 节点数降为原来的1/4

    x_flat = GraphConvLayer(filters)(x_flat)
    x_flat = Activation('relu')(x_flat)

    x_reshaped = Reshape((h, w, filters))(x_flat)
    # 上采样恢复尺寸，需匹配新增的池化层次数
    x_upsampled = UpSampling2D(size=(2, 2), interpolation='bilinear')(x_reshaped)
    x_upsampled = UpSampling2D(size=(2, 2), interpolation='bilinear')(x_upsampled)  # 新增的上采样

    explanation = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x_upsampled)
    return explanation


# -------------------------------
# 12. 图像解释网络
# -------------------------------
def image_explanation_network(inputs, segmentation):
    """
    结合原始输入与分割结果，通过卷积网络生成图像解释图
    """
    x = Concatenate()([inputs, segmentation])
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    return x


# -------------------------------
# 13. 整体模型构建
# -------------------------------
def build_model(input_shape):
    inputs = Input(input_shape)

    # 第一分支：基于 VGG19 的编码器1
    x1, skip_1 = encoder1(inputs)
    # 插入多尺度注意力模块
    # x1 = multi_scale_attention_block(x1)
    x1 = lightweight_aspp(x1, 64)
    x1 = decoder1(x1, skip_1)
    outputs1 = output_block(x1)

    # 第二分支：将原图与第一分支输出进行元素级相乘
    x2 = Multiply()([inputs, outputs1])
    x2, skip_2 = encoder2(x2)
    # 插入多实例学习与自注意力模块
    x2 = multi_instance_self_attention_block(x2)
    x2 = lightweight_aspp(x2, 64)
    x2 = decoder2(x2, skip_1, skip_2)
    # 保存解码器第二分支的特征供解释性模块使用
    x2_features = x2
    outputs2 = output_block(x2_features)

    # 融合第一分支与第二分支的输出得到最终分割结果
    combined = Concatenate()([outputs1, outputs2])
    # segmentation = Conv2D(1, (1, 1), activation='sigmoid')(combined)
    segmentation = Conv2D(1, (1, 1), activation='sigmoid', name="segmentation")(combined)

    # 解释性分支1：基于图卷积的解释模块，作用于第二分支解码器特征
    graph_explanation = graph_conv_explain_block(x2_features)
    # 解释性分支2：图像解释网络，结合原图与分割结果
    img_explanation = image_explanation_network(inputs, segmentation)
    # 融合两路解释信息
    explanation = Concatenate()([graph_explanation, img_explanation])
    # explanation = Conv2D(1, (1, 1), activation='sigmoid')(explanation)
    explanation = Conv2D(1, (1, 1), activation='sigmoid', name="explanation")(explanation)

    # 最终模型输出：分割结果与解释结果（方便后续分析）
    model = Model(inputs, [segmentation, explanation])
    return model


# -------------------------------
# 14. 主程序：模型构建、概述与可视化
# -------------------------------
if __name__ == "__main__":
    model = build_model((384, 512, 3))
    model.summary()
    try:
        plot_model(
            model,
            to_file='model_structure1.png',
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=96  # 设置图像分辨率
        )
        print("模型架构图已成功保存为 'model_structure1.png'")
    except Exception as e:
        print("绘制模型架构图时出错：", e)
