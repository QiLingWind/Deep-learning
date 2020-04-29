import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
datapath  = r'E:\Pycharm\project\project_TF\.idea\data\mnist.npz'
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(datapath)
x_test1 = x_test    #用于打印图片28*28*1
x_train = x_train.reshape(x_train.shape[0],784).astype('float32')
x_test = x_test.reshape(x_test.shape[0],784).astype('float32')


x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)   #归一化

model = tf.keras.Sequential([ # 3 个非线性层的嵌套模型
    tf.keras.layers.Flatten(),  #将多维数据打平,也即reshape为60000*784，否则就要在前面先reshape，而reshape之后就不用这个了
    tf.keras.layers.Dense(500, activation='relu'),#128
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
# 打印模型
model.build((None,784,1))
print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
model.fit(x_train, y_train, epochs=1,verbose=2) #verbose为1表示显示训练过程

#这里是预测模型
val_loss, val_acc = model.evaluate(x_test, y_test) # model.evaluate是输出计算的损失和精确度
print('First Test Loss:{:.6f}'.format(val_loss))
print('First Test Acc:{:.6f}'.format(val_acc))

#预测模型方式二
acc_correct = 0
predictions = model.predict(x_test)     # model.perdict是输出预测结果

for i in range(len(x_test)):
    if (np.argmax(predictions[i]) == y_test[i]):    # argmax是取最大数的索引,放这里是最可能的预测结果
        acc_correct += 1

print('Test accuracy:{:.6f}'.format(acc_correct*1.0/(len(x_test))))

i = 0
plt.imshow(x_test1[i],cmap=plt.cm.binary)
plt.show()
# predictions = model.predict(x_test)
print(np.argmax(predictions[i]))    # argmax输出的是最大数的索引，predicts[i]是十个分类的权值
print((predictions[i]))             # 比如predicts[0]最大的权值是第八个数，索引为7，故预测的数字为7
