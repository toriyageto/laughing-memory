import struct
import numpy as np
import tensorflow as tf
from keras.losses import categorical_crossentropy
import keras

def decode_idx3_ubyte(idx3_ubyte_file):  # 此函数用来解析idx3文件，idx3_ubyte_filec指定图像文件路径
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offest = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offest)
    # print('魔数：%d,图片数量：%d，图片大小：%d%d' % (magic_number,num_images,num_rows,num_cols))
    # 解析数据集
    image_size = num_rows * num_cols
    offest += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
        #     print('已解析%d'%(i+1)+'张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offest)).reshape((num_rows, num_cols))
        offest += struct.calcsize(fmt_image)
    return images


'''images是一个三维数组,images[i][a][b]表示第i张图片的倒数第a行，b列的像素'''


def decode_idx1_ubyte(idx1_ubyte_file):  # 解析idx1文件函数，idx1_ubyte_file指定标签文件路径
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数和标签数
    offest = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offest)
    # print('魔数：%d，图片数量：%d张' % (magic_number,num_images))
    # 解析数据集
    offest += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
        #     print('已解析：%d'%(i+1)+'张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offest)[0]
        offest += struct.calcsize(fmt_image)
    # print(labels[0])
    return labels


def CreateDatasets(high, iNum, iArraySize):
    x_train = np.random.random((iNum, iArraySize)) * float(high)
    y_train = ((x_train[:iNum, 0] + x_train[:iNum, 1]) / 2).astype(int)
    return x_train, y_train, tf.keras.utils.to_categorical(y_train, num_classes=high)


images = decode_idx3_ubyte('train-images.idx3-ubyte')
labels = decode_idx1_ubyte('train-labels.idx1-ubyte')
images_train, images_test, labels_train, labels_test = images[:50000], images[-500:], labels[:50000], labels[-500:]
# images_train_2d = images_train.reshape(50000, 784)
# images_test_2d = images_test.reshape(500, 784)
images_train_3d=images_train.reshape(50000,28,28,1)
images_test_3d=images_test.reshape(500,28,28,1)
# print(images_train_3d.shape)

# 标准化
images_train_3d = images_train_3d.astype(float)
images_test_3d = images_test_3d.astype(float)
images_train_3d /= 255
images_test_3d /= 255
# print(images_test_2d)

# 独热编码
category = 10  # 0-9共10个标签
labels_train2 = keras.utils.to_categorical(labels_train, num_classes=category)
labels_test2 = keras.utils.to_categorical(labels_test, num_classes=category)
# print(labels_train2)

# # 创建MLP模型
# dim = 784
# model = keras.models.Sequential()
# model.add(keras.layers.Dense(units=10, activation=tf.nn.relu, input_dim=dim))
# model.add(keras.layers.Dense(units=10, activation=tf.nn.relu))
# model.add(keras.layers.Dense(units=10, activation=tf.nn.relu))
# model.add(keras.layers.Dense(units=category, activation=tf.nn.softmax))
# 创建CNN模型
model=keras.models.Sequential()
model.add(keras.layers.Conv2D(
    filters=3,   #1张图片变3个图片
    kernel_size=(3,3), #卷积层滤镜尺寸
    padding='same',  #相同尺寸
    activation=tf.nn.relu, #relu激活函数
    input_shape=(28,28,1) #原灰度图的尺寸
))
model.add(keras.layers.MaxPool2D(pool_size=(2,2))) #宽和高各缩小一半
model.add(keras.layers.Conv2D(
    filters=9,
    kernel_size=(2,2),
    padding='same',
    activation=tf.nn.relu
))
model.add(keras.layers.Dropout(rate=0.5)) #随机丢失一半图片
model.add(keras.layers.Conv2D(
    filters=6,
    kernel_size=(2,2),
    padding='same',
    activation=tf.nn.relu
))
model.add(keras.layers.Flatten())  #传送至MLP模型，变成一维数据
model.add(keras.layers.Dense(units=10, activation=tf.nn.relu))
model.add(keras.layers.Dense(units=category, activation=tf.nn.softmax))
# model.summary()

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=categorical_crossentropy, metrics=['accuracy'])
# model.summary()

model.fit(x=images_train_3d, y=labels_train2, epochs=100, batch_size=1000, verbose=1)

score = model.evaluate(images_test_3d, labels_test2, batch_size=128)
print(score)
# MLP [0.2566913664340973, 0.9599999785423279]
# CNN [0.1361602246761322, 0.9800000190734863]

# 测试前10个测试集与预测值是否一致
predict = model.predict(images_test_3d[:10])
for i in range(len(labels_test2[:10])):
    print(np.argmax(predict[i]), end=' ')
print()
print(labels_test[:10])
