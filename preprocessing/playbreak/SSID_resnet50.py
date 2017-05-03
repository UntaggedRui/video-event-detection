from keras.preprocessing.image import ImageDataGenerator
import numpy as np


from keras.models import Model
from keras.preprocessing import image as keras_image
import os,sys

model_path = os.path.join('..','models','keras','models')
sys.path.append(model_path)

import resnet50

weights_path = 'weights_resnet50.h5'
weights_path = os.path.abspath(weights_path)
model = resnet50.ResNet50(weights_path=weights_path)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data

ssid_root = os.path.join('..','dataset','SSID_Dataset')

train_path = os.path.join(ssid_root,'train')
train_generator = train_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 256x256
        batch_size=32,
        class_mode='categorical')

# this is a similar generator, for validation data
val_path = os.path.join(ssid_root,'validation')
validation_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)

model.save_weights(weights_path)  # always save your weights after training or during training

files = os.listdir('test-image')

index = 0

feature_model = Model(input=model.input, output=model.get_layer('avg_pool').output)
for file in files:
    img = keras_image.load_img('test-image/' + file, target_size=(224, 224))
    rescale = 1. / 255
    x = keras_image.img_to_array(img, dim_ordering='tf')
    x = np.expand_dims(x, axis=0)
    x = x * rescale
    #x = preprocess_input(x)
    result = feature_model.predict(x)
    #result = np.argmax(result)
    print result, file, index/3+1
    index += 1


