import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

#导入数据集并进行处理


(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]','train[80%:90%]','train[90%:]'],
    shuffle_files=True,
    batch_size=None,
    with_info=True,
    as_supervised=True,
)
IMG_SIZE = 160  #ALL

def format_example(image,label):
    image = tf.image.resize(image,(IMG_SIZE,IMG_SIZE))
    return image,label

train =  raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
train_bathes = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

#导入MobileNetV2模型
IMG_SHAPE = (IMG_SIZE,IMG_SIZE,3)

# Create model
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
# 展示模型
base_model.summary()

#添加自己的层
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1)
])
# 展示最终模型
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

#训练模型
history = model.fit(train_bathes,
                    epochs=10,
                    validation_data=validation_batches)


















