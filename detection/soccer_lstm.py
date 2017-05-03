from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM
import scipy.io as sio
import os,sys
from keras.optimizers import SGD


def lstm(input_shape=None, nb_classes=0):
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    # Model.
    model = Sequential()
    model.add(LSTM(2048, return_sequences=True, input_shape=input_shape,
                   dropout_W=0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    return model


batch_size = 64

print('Loading data...')
path='val_train256.mat'
data=sio.loadmat(path)
x_train=data['x_train']
y_train=data['y_train']
x_train=x_train.astype('float32')
y_train=y_train.astype('float32')
x_val=data['x_val']
y_val=data['y_val']
x_val=x_val.astype('float32')
y_val=y_val.astype('float32')

print('x_train shape:', x_train.shape)
y_train=y_train.tolist();
y_val=y_val.tolist();
print('Build model...')

model = lstm(input_shape=(256,512),nb_classes=5)

weights_path = 'lstm_weights'
if os.path.exists('weights_path'):
    model.load_weights(weights_path)
# try using different optimizers and different optimizer configs
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical',metrics=['accuracy'])

print("Train...")
model.fit(x_train, y_train, batch_size=batch_size,validation_data=(x_val, y_val), nb_epoch=150)
score, acc = model.evaluate(x_val, y_val,
                            batch_size=batch_size)
#json_string=model.to_json()
#open('soccer_model_architecture.json','w').write(json_string)
#model.save_weights('soccer_model_weights.h5')

#from keras.modelsimport model_from_json
#model = model_from_json(open('my_model_architecture.json').read())
#model.load_weights('my_model_weights.h5')
model.save_weights('lstm_weights.h5')
y_dim = model.predict(x_val, batch_size=batch_size, verbose=1)
y_pre = model.predict_classes(x_val, batch_size=batch_size, verbose=1)

print('Test score:', score)
print('Test accuracy:', acc)



