import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

datapath  = r'E:\Pycharm\project\project_TF\.idea\data\mnist.npz'
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(datapath)
#数据预处理 导入的为60000*28*28 的数据，全连接网络应reshape为60000*784（也即打平Flatten()）， 卷积网络则为60000*28*28*1（黑白单通道）
x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')
# 归一化
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)   #归一化

model = keras.models.Sequential([
    layers.Conv2D(filters=16, kernel_size=(5,5), padding='same',#卷积核个数为16，same代表输出图像和输入图像大小一致
                 input_shape=(28,28,1),  activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=36, kernel_size=(5,5), padding='same',
    			 activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10,activation='softmax')
])
#打印模型
print(model.summary())
#训练配置
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
#开始训练
model.fit(x=x_train, y=y_train, validation_split=0.2,   #validation_split是指将一部分作为测试集使用，0.2代表80%的数据作为训练集
                        epochs=10,batch_size=128, verbose=1)   #verbose=1代表显示训练过程
val_loss, val_acc = model.evaluate(x_test, y_test)
print('Test Loss:{:.6f}'.format(val_loss))
print('Test Acc:{:.6f}'.format(val_acc))

