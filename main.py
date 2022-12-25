import os
import sys
from random import shuffle

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications import VGG16
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Model, Sequential, load_model, save_model

# 视频目录
in_dir = "./video"
# 图片大小尺寸，也即帧大小。224 * 224
img_size = 224
img_size_tuple = (img_size, img_size)
# 表示图片为RGB三通道
num_channels = 3
# 224 * 224 * 3
img_size_flat = img_size * img_size * num_channels
# 暴力和非暴力两类
num_classes = 2
# Number of files to train
num_files_train = 1
# 每个视频20帧
images_per_file = 20
# 每个训练集的帧数
num_images_train = num_files_train * images_per_file


# 输出处理训练集和测试集的进度
def print_progress(count, max_count):
    # 完成百分比.
    pct_complete = count / max_count

    # 状态消息。请注意\r意味着该行应覆盖自身
    msg = "\r- Progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


# 获取视频帧的get_frames函数，函数用于从视频文件中获取20帧，并将帧转换为适合神经网络的格式。
def get_frames(current_dir, file_name):
    in_file = os.path.join(current_dir, file_name)

    images = []

    vidcap = cv2.VideoCapture(in_file)

    success, image = vidcap.read()

    count = 0

    while count < images_per_file:
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        res = cv2.resize(RGB_img, dsize=(img_size, img_size),
                         interpolation=cv2.INTER_CUBIC)

        images.append(res)

        success, image = vidcap.read()

        count += 1

    ret = np.array(images)

    ret = (ret / 255.).astype(np.float16)

    return ret


# 获取数据的名称并对其进行标记：（如果视频名字带V则为暴力视频，如果NV则为非暴力视频）
def label_video_names(in_dir):
    # list containing video names
    names = []
    # list containing video labels [1, 0] if it has violence and [0, 1] if not
    labels = []

    for current_dir, dir_names, file_names in os.walk(in_dir):

        for file_name in file_names:
            # 标记暴力和非暴力视频
            if file_name[0:1] == 'V':
                labels.append([1, 0])
                names.append(file_name)
            elif file_name[0:2] == 'NV':
                labels.append([0, 1])
                names.append(file_name)

    c = list(zip(names, labels))
    # Suffle the data (names and labels)
    shuffle(c)

    names, labels = zip(*c)

    return names, labels


# 函数通过VGG16处理20个视频帧并获得传输值
def process_transfer(vid_names, in_dir, labels):
    count = 0

    # vid_names表示训练集或测试集的视频个数
    length = len(vid_names)

    # 将数据改为20，224，224，3 以符合VGG16的输入
    shape = (images_per_file,) + img_size_tuple + (3,)

    while count < length:
        video_name = vid_names[count]

        image_batch = get_frames(in_dir, video_name)

        # Note that we use 16-bit floating-points to save memory.
        shape = (images_per_file, transfer_values_size)

        transfer_values = image_model_transfer.predict(image_batch)

        labels1 = labels[count]

        aux = np.ones([20, 2])

        labelss = labels1 * aux

        yield transfer_values, labelss

        count += 1


# 用于保存VGG16的传输值以供以后使用的函数，保存至chunkTrain.h5文件
def make_files_train(n_files):
    gen = process_transfer(names_train, in_dir, labels_train)

    num = 1

    # Read the first chunk to get the column dtypes
    chunk = next(gen)

    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]

    with h5py.File('chunkTrain.h5', 'w') as f:
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]

        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)

        # Write the first chunk of rows
        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:
            if num == n_files:
                break

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            # Increment the row count
            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]

            print_progress(num, n_files)

            num += 1


# 保存至chunkTest.h5文件
def make_files_test(n_files):
    gen = process_transfer(names_test, in_dir, labels_test)

    numer = 1

    # Read the first chunk to get the column dtypes
    chunk = next(gen)

    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]

    with h5py.File('chunkTest.h5', 'w') as f:
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]

        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)

        # Write the first chunk of rows
        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:
            if numer == n_files:
                break

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            # Increment the row count
            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]

            print_progress(numer, n_files)

            numer += 1


# 为了将保存的传输值加载到RAM内存中，使用以下两个函数：
def process_alldata_train():
    joint_transfer = []
    frames_num = 20
    count = 0

    with h5py.File('chunkTrain.h5', 'r') as f:
        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch) / frames_num)):
        inc = count + frames_num
        joint_transfer.append([X_batch[count:inc], y_batch[count]])
        count = inc

    data = []
    target = []

    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))

    return data, target


def process_alldata_test():
    joint_transfer = []
    frames_num = 20
    count = 0

    with h5py.File('chunkTest.h5', 'r') as f:
        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch) / frames_num)):
        inc = count + frames_num
        joint_transfer.append([X_batch[count:inc], y_batch[count]])
        count = inc

    data = []
    target = []

    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))

    return data, target


# 首先得到整个视频的名称和标签
names, labels = label_video_names(in_dir)
# 测试视频文件名是否正确
# print("测试视频文件名是否正确：" + names[12])

frames = get_frames(in_dir, names[12])

# 将帧转换回uint8像素格式以打印帧
# visible_frame = (frames*255).astype('uint8')
# plt.imshow(visible_frame[3])

# 预训练模型VGG16
image_model = VGG16(include_top=True, weights='imagenet')

# 输出模型layer说明
# image_model.summary()

# 用VGG16模型批量输入和处理20帧视频。
# 当所有的视频经过VGG16模型处理后，得到的传输值保存到一个缓存文件中，我们就可以将这些传输值作为LSTM神经网络的输入。
# 然后，我们将使用暴力数据集（暴力，无暴力）中的类来训练第二个神经网络，以便该网络学习如何基于VGG16模型中的传递值对图像进行分类。
# 模型输入为224×224×3。其中224位帧大小，3为RGB通道
# 我们将在最后的分类层fc2之前使用该层的输出。fc2层为全连接层
transfer_layer = image_model.get_layer('fc2')

image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)

transfer_values_size = K.int_shape(transfer_layer.output)[1]

# 划分数据集
train_len = int(len(names) * 0.8)
test_len = int(len(names) * 0.2)

names_train = names[0:train_len]
names_test = names[train_len:]

labels_train = labels[0:train_len]
labels_test = labels[train_len:]

# 然后我们将通过VGG16处理所有视频帧并保存传输值
print(make_files_train(train_len))
print(make_files_test(test_len))

data, target = process_alldata_train()
data_test, target_test = process_alldata_test()

# 定义LSTM体系结构
# VGG16网络从每个帧获得4096个传输值的向量作为输出。
# 从每个视频，我们正在处理20帧，所以我们将有20 x 4096每个视频值。
# 分类必须考虑到视频的20帧。如果他们中的任何一个检测到暴力，视频将被归类为暴力。
chunk_size = 4096
n_chunks = 20
rnn_size = 512

# 搭建模型
model = Sequential()
# LSTM处理图片序列
model.add(LSTM(rnn_size, input_shape=(n_chunks, chunk_size)))
# 数字代表输出维度
model.add(Dense(1024))
# 激活函数relu
model.add(Activation('relu'))
model.add(Dense(50))
# 激活函数sigmoid
model.add(Activation('sigmoid'))
model.add(Dense(2))
# 最后一层为softmax
model.add(Activation('softmax'))
# 优化器adam
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# 训练模型
epoch = 200
batchSize = 500

history = model.fit(np.array(data[0:1200]), np.array(target[0:1200]), epochs=epoch,
                    validation_data=(np.array(data[1200:]), np.array(target[1200:])),
                    batch_size=batchSize, verbose=2)

# 保存模型
model.save('vd.hdf5')

# 读取模型并预测
# model = load_model("./model.hdf5")
# cost, accuracy = model.evaluate(X_test,Y_test)
# print("accuracy: ",accuracy)

# 测试模型
result = model.evaluate(np.array(data_test), np.array(target_test))

# 打印模型精度
for name, value in zip(model.metrics_names, result):
    print(name, value)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# 保存文件
plt.savefig('accuracy.eps', format='eps', dpi=1000)
plt.show()

# 损失函数
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# 保存文件
plt.savefig('loss.eps', format='eps', dpi=1000)
plt.show()
