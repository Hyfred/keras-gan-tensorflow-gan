# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 18:08:45 2019

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 09:51:49 2019

@author: hyfred
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

#合并画出数据的命令
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

#def plot_image(image):
#    fig = plt.gcf()
#    fig.set_size_inches(2, 2) #设置图形大小
#    plt.imshow(image, cmap='binary') #以黑白灰度显示
#    plt.show()


(X_train, y_train), (X_test, y_test) = mnist.load_data() #获取数据集
#print(X_train.shape)
#plot_image(X_train[0]) #画出第1张图
#print(y_train[0])
X_train = (X_train.astype(np.float32) - 127.5)/127.5#对数据中心化处理，或许可以直接标准化除以255

X_train = X_train[:, :, :, None]
X_test = X_test[:, :, :, None]#如果是拉伸成一维就是X_train.reshape(60000, 784).asytpe('float32')
#X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) #这种方法与上面一致

########构建判别器#######
d = Sequential()
d.add(Conv2D(64, (5, 5),
                 padding='same',
                 input_shape=(28, 28, 1)))
d.add(Activation('tanh'))
d.add(MaxPooling2D(pool_size=(2, 2)))
d.add(Conv2D(128, (5, 5)))#这里没有填充，故14-5+1=10
d.add(Activation('tanh'))
d.add(MaxPooling2D(pool_size=(2, 2)))
d.add(Flatten())
d.add(Dense(1024))
d.add(Activation('tanh'))
d.add(Dense(1))
d.add(Activation('sigmoid'))
#查看判别器模型
print(d.summary())

########构建生成器#######
g = Sequential()
g.add(Dense(units=1024, input_dim=100))
g.add(Activation('tanh'))
g.add(Dense(128*7*7))
g.add(BatchNormalization())
g.add(Activation('tanh'))
g.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
g.add(UpSampling2D(size=(2, 2)))
g.add(Conv2D(64, (5, 5), padding='same'))
g.add(Activation('tanh'))
g.add(UpSampling2D(size=(2, 2)))
g.add(Conv2D(1, (5, 5), padding='same'))
g.add(Activation('tanh'))
#查看生成器模型
print(g.summary())

#两个模型连接在一起
model = Sequential()
model.add(g)
d.trainable = False
model.add(d)
#查看总模型
print(model.summary())

BATCH_SIZE=128#一批大小（即一批多少个）

#对构建的网络进行设置
d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
g.compile(loss='binary_crossentropy', optimizer="SGD")
model.compile(loss='binary_crossentropy', optimizer=g_optim)
d.trainable = True
d.compile(loss='binary_crossentropy', optimizer=d_optim)

#训练5个周期
for epoch in range(5):
    print("Epoch is", epoch) #打印这是第几个训练周期
    print("Number of batches", int(X_train.shape[0]/BATCH_SIZE)) #打印训练多少个批次
    #一个周期训练多少批（等于总数除以一批的数量）
    for index in range(int(X_train.shape[0]/BATCH_SIZE)):
        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
        image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE] #获得一批的真实图片
        generated_images = g.predict(noise, verbose=0) #生成一批虚假图片
        # 每隔多少次生成一次图片，并保存成png格式
        if index % 20 == 0:
            image = combine_images(generated_images)
            image = image*127.5+127.5
            Image.fromarray(image.astype(np.uint8)).save(
                str(epoch)+"_"+str(index)+".png")
        X = np.concatenate((image_batch, generated_images)) #混合真实图片和虚假图片
        y = [1] * BATCH_SIZE + [0] * BATCH_SIZE #前一批真实图片的标签为1，后一批虚假图片的标签为0
        d_loss = d.train_on_batch(X, y) #训练判别器D
        print("batch %d d_loss : %f" % (index, d_loss))
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100)) #生成服从正态分布的噪音
        d.trainable = False #固定判别器D的参数
        g_loss = model.train_on_batch(noise, [1] * BATCH_SIZE) #训练生成器G
        d.trainable = True #打开判别器D的参数，为了新一轮的循环
        print("batch %d g_loss : %f" % (index, g_loss)) #打印批次和损失
        #隔多少次保存一次模型
        if index % 10 == 9:
            g.save_weights('generator', True)
            d.save_weights('discriminator', True)

#取出生成器来生成数字
def generate(BATCH_SIZE, nice=False):
    g = Sequential()
    g.add(Dense(units=1024, input_dim=100))
    g.add(Activation('tanh'))
    g.add(Dense(128*7*7))
    g.add(BatchNormalization())
    g.add(Activation('tanh'))
    g.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    g.add(UpSampling2D(size=(2, 2)))
    g.add(Conv2D(64, (5, 5), padding='same'))
    g.add(Activation('tanh'))
    g.add(UpSampling2D(size=(2, 2)))
    g.add(Conv2D(1, (5, 5), padding='same'))
    g.add(Activation('tanh'))
    
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = Sequential()
        d.add(Conv2D(64, (5, 5),
                         padding='same',
                         input_shape=(28, 28, 1)))
        d.add(Activation('tanh'))
        d.add(MaxPooling2D(pool_size=(2, 2)))
        d.add(Conv2D(128, (5, 5)))#这里没有填充，故14-5+1=10
        d.add(Activation('tanh'))
        d.add(MaxPooling2D(pool_size=(2, 2)))
        d.add(Flatten())
        d.add(Dense(1024))
        d.add(Activation('tanh'))
        d.add(Dense(1))
        d.add(Activation('sigmoid'))

        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")
