import os
import numpy as np
import glob
import shutil

import tensorflow as tf

import matplotlib.pyplot as plt

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    print("{}: {} Images".format(cl, len(images)))
    train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]

    for t in train:
        if not os.path.exists(os.path.join(base_dir, 'train', cl)):
            os.makedirs(os.path.join(base_dir, 'train', cl))
        shutil.move(t, os.path.join(base_dir, 'train', cl))

    for v in val:
        if not os.path.exists(os.path.join(base_dir, 'val', cl)):
            os.makedirs(os.path.join(base_dir, 'val', cl))
        shutil.move(v, os.path.join(base_dir, 'val', cl))

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

batch_size = 100;   #100 images per each training step
img_shape = 150;    #150x150 images in flower dataset

model = tf.keras.Sequential();

train_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 45, zoom_range = 0.5, horizontal_flip = True, width_shift_range= 0.15, height_shift_range=0.15, rescale = ((1.0)/255) );
valid_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = ((1.0)/255) );

train_images = train_image_gen.flow_from_directory(train_dir, batch_size= batch_size, target_size=(img_shape,img_shape), shuffle=True, class_mode = "sparse");
valid_images = valid_image_gen.flow_from_directory(val_dir, batch_size=batch_size, target_size=(img_shape,img_shape), class_mode = "sparse");

conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation=tf.keras.activations.relu, input_shape=(img_shape, img_shape, 3));
mp_1 = tf.keras.layers.MaxPooling2D( pool_size=(2,2), padding="same");
model.add(conv_1);
model.add(mp_1);

conv_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation=tf.keras.activations.relu);
mp_2 = tf.keras.layers.MaxPooling2D( pool_size=(2,2), padding="same");
model.add(conv_2);
model.add(mp_2);

conv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation=tf.keras.activations.relu);
mp_3 = tf.keras.layers.MaxPooling2D( pool_size=(2,2), padding="same");
model.add(conv_3);
model.add(mp_3);

flatten_1 = tf.keras.layers.Flatten();
fcn_1 = tf.keras.layers.Dense(units = 512, activation=tf.keras.activations.softmax);
model.add(flatten_1);
model.add(fcn_1);

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]);

epochs = 80;

history = model.fit_generator(generator=train_images, epochs = epochs, verbose = 2, validation_data=valid_images)
