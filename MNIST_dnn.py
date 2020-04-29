import os
import tensorflow as tf # 导入 TF 库
from tensorflow import keras # 导入 TF 子库
from tensorflow.keras import layers, optimizers, datasets # 导入 TF 子库
(x, y), (x_val, y_val) = datasets.mnist.load_data() # 加载数据集
x = 2*tf.convert_to_tensor(x, dtype=tf.float32)/255.-1 # 转换为张量，缩放到-1~1
y = tf.convert_to_tensor(y, dtype=tf.int32) # 转换为张量
y = tf.one_hot(y, depth=10) # one-hot 编码
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y)) # 构建数据集对象
train_dataset = train_dataset.batch(512) # 批量训练

model = keras.Sequential([ # 3 个非线性层的嵌套模型
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')])

with tf.GradientTape() as tape: # 构建梯度记录环境
    # 打平，[b, 28, 28] => [b, 784]
    x = tf.reshape(x, (-1, 28 * 28))    #
    # Step1. 得到模型输出 output
    # [b, 784] => [b, 10]
    out = model(x)
print(model.summary())
# Step3. 计算参数的梯度 w1, w2, w3, b1, b2, b3
#grads = tape.gradient(loss, model.trainable_variables)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
#开始训练
model.fit(x=x, y=y, validation_split=0.2,epochs=10, batch_size=50, verbose=2)






