from keras import backend as K

if K.backend() == "tensorflow":
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.80
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))
    print("Setting {}% of the GPU memory ...".format(config.gpu_options.per_process_gpu_memory_fraction * 100))

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, CuDNNLSTM, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from util import Helper
import re
import os


helper = Helper()
train_data = helper.get_train_desc()
train_len = train_data.shape[0]
print(train_len)
test_data = helper.get_test_desc()
test_len = test_data.shape[0]
print(test_len)
alpha_numeric_filter = lambda x: re.sub('[^a-zA-z0-9\s]','',x)
train_data = train_data.apply(alpha_numeric_filter)
test_data = test_data.apply(alpha_numeric_filter)

max_features = 2000
tokenizer = Tokenizer(num_words = max_features, split = ' ')
data = train_data.append(test_data)
tokenizer.fit_on_texts(data.values)
data = tokenizer.texts_to_sequences(data.values)
data = pad_sequences(data)
print(data.shape)
train_data = data[:train_len]
test_data = data[train_len:]
print(train_data.shape)
print(test_data.shape)
embed_out = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_out, input_length = train_data.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(CuDNNLSTM(lstm_out)))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

fname = os.path.join("models", "weights.{epoch:02d}-{val_acc:.2f}.hdf5")
checkpoint = ModelCheckpoint(fname, monitor = "val_acc", mode = "max", save_best_only = True, verbose = 1)
callbacks = [checkpoint]
target = helper.get_target()
encoder = LabelEncoder()
target = encoder.fit_transform(target)
print(encoder.classes_)
X_train, X_val, Y_train, Y_val = train_test_split(train_data, target, test_size = 0.3, random_state = 42, stratify = target)
print(X_train[:10])
print(Y_train[:10])
batch_size = 32
model.fit(X_train, Y_train, epochs = 10, validation_data = (X_val, Y_val), batch_size = batch_size, verbose = 1)

model.save("models/lstm.h5")
pred = model.predict(test_data)
pred = np.ravel(pred)
pred = (np.array(pred) > 0.5).astype(np.int)
pred = encoder.inverse_transform(pred)
helper.generate_result(pred, suffix = "lstm")

