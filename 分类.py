import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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


images = decode_idx3_ubyte('train-images.idx3-ubyte')
# print(images.shape)
labels = decode_idx1_ubyte('train-labels.idx1-ubyte')
# print(labels)

# 样例图片展示
some_digit_image=images[50000]
print(labels[50000])
# print(some_digit_image)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis('off')
plt.show()

images_train, images_test, labels_train, labels_test = images[:5000], images[-500:], labels[:5000], labels[-500:]
images_train_2d=images_train.reshape(5000,784)
# 多分类器
sgd_clf=SGDClassifier(random_state=42)
sgd_clf.fit(images_train_2d,labels_train)
# print(some_digit_image_2d)
some_digit_image_2d=some_digit_image.reshape(1,784)
# predictions=sgd_clf.predict(some_digit_image_2d)
# print(predictions) #与上文的print(labels[55000])相比较，可以看到基本一致
# images_test_2d=images_test.reshape(500,784)
# predictions=sgd_clf.predict(images_test_2d)
# mse = mean_squared_error(labels_test, predictions)
# rmse = np.sqrt(mse)
# print(rmse) 1.1153474794878948，均方根误差非常低，模型还可以

# score=cross_val_score(sgd_clf, images_train_2d,labels_train, cv=5, scoring="accuracy")
# print(score)

# 正则化
# scaler = StandardScaler()
# images_train_2d_scaled = scaler.fit_transform(images_train_2d.astype(np.float64))
# score=cross_val_score(sgd_clf, images_train_2d_scaled, labels_train, cv=5, scoring="accuracy")
# print(score)