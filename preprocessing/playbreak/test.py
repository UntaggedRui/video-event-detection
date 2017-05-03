import os,sys
import numpy as np
from keras.models import Model
from keras.preprocessing import image as keras_image

model_path = os.path.join('..','models','keras','models')
sys.path.append(model_path)

import resnet50

if __name__ == '__main__':
    weights_path = 'weights_resnet50.h5'
    weights_path = os.path.abspath(weights_path)
    model = resnet50.ResNet50(weights_path=weights_path)

    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    files = os.listdir('test-image')
    files.sort()
    feature_model = Model(input=model.input, output=model.get_layer('avg_pool').output)
    for file in files:
        img = keras_image.load_img('test-image/' + file, target_size=(224, 224))

        x = keras_image.img_to_array(img, dim_ordering='tf')
        rescale = 1. / 255
        x = x*rescale
        x = np.expand_dims(x, axis=0)

        result = model.predict(x)
        # result = feature_model.predict(x)
        result = np.argmax(result)
        print result, file
