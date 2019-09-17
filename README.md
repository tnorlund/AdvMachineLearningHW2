

```python
import keras
from keras.layers import Input, Conv2D, Flatten, MaxPool2D, Dropout, Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from keras import models
from keras.models import Model
from imgaug import augmenters
from random import randint
import pandas as pd
import numpy as np
import cv2
import time
```

# Problem 1

Choose a small (< 3,000) image dataset for classification. Include the link where you have downloaded the pictures from.

## 1

Train a model from scratch using what little data you have without any regularization, to set a baseline for what can be achieved.

DOWNLOAD HERE:

https://www.kaggle.com/zalando-research/fashionmnist/version/4

---



```python
epochs = 30
batch_size = 256
verbose = 1
```


```python
train = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")
trainX = train[list(train.columns)[1:]].values
trainY = to_categorical(train['label'].values)
testX = train[list(train.columns)[1:]].values
testY = to_categorical(train['label'].values)
```

*Train-Validate Split*


```python
trainX, valX, trainY, valY = train_test_split(
    trainX,
    trainY, 
    test_size=0.2,
    random_state=18
)
```

*Normalization*


```python
trainX = trainX/255.
testX = testX/255.
valX = valX/255.
```

*Reshape*


```python
trainX = trainX.reshape(-1, 28, 28, 1)
testX = testX.reshape(-1, 28, 28, 1)
valX = valX.reshape(-1, 28, 28, 1)
```

*Model*


```python
# input
input_layer = Input(shape=(28, 28, 1))

# conv layers
conv_layer1   = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
conv_layer1   = MaxPool2D( (2, 2), padding='same')(conv_layer1)
conv_layer1   = Dropout(0.25)(conv_layer1)

conv_layer2   = Conv2D(64, (3, 3), activation='relu')(conv_layer1)
conv_layer2   = MaxPool2D( (2, 2), padding='same')(conv_layer2)
conv_layer2   = Dropout(0.25)(conv_layer2)

conv_layer3   = Conv2D(128, (3, 3), activation='relu')(conv_layer2)
conv_layer3   = Dropout(0.4)(conv_layer3)

# flatten and dense layers
flatten_layer = Flatten()(conv_layer3)
dense_layer   = Dense(128, activation='relu')(flatten_layer)
dense_layer   = Dropout(0.3)(dense_layer)

# output
output_layer  = Dense(10, activation='softmax')(dense_layer)

model = Model(input_layer, output_layer)
model.compile(
    optimizer=keras.optimizers.Adam(), 
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy'])
if verbose==1:
    print(model.summary())
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 28, 28, 1)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 28, 28, 32)        320       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 12, 12, 64)        18496     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 6, 6, 64)          0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 4, 4, 128)         73856     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 4, 4, 128)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 2048)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               262272    
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 356,234
    Trainable params: 356,234
    Non-trainable params: 0
    _________________________________________________________________
    None


*Training*


```python
tic = time.clock()

early_stopping = EarlyStopping(
    monitor='val_loss', 
    min_delta=0, 
    patience=10, 
    verbose=verbose, 
    mode='auto')

history = model.fit(
    trainX, 
    trainY, 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=verbose,
    validation_data=(valX, valY), 
    callbacks=[early_stopping])

toc = time.clock()
print(f"Conv Model took {toc-tic} seconds")

score = model.evaluate(testX, testY, verbose=0)
print('Test loss:    ', score[0])
print('Test accuracy:', score[1])
predictions = model.predict(testX)
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/30
    48000/48000 [==============================] - 50s 1ms/step - loss: 0.7914 - acc: 0.6999 - val_loss: 0.4500 - val_acc: 0.8372
    Epoch 2/30
    48000/48000 [==============================] - 49s 1ms/step - loss: 0.4767 - acc: 0.8246 - val_loss: 0.3729 - val_acc: 0.8661
    Epoch 3/30
     5888/48000 [==>...........................] - ETA: 33s - loss: 0.4016 - acc: 0.8482


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-9-dc35c71ec85e> in <module>
         15     verbose=verbose,
         16     validation_data=(valX, valY),
    ---> 17     callbacks=[early_stopping])
         18 
         19 toc = time.clock()


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
       1176                                         steps_per_epoch=steps_per_epoch,
       1177                                         validation_steps=validation_steps,
    -> 1178                                         validation_freq=validation_freq)
       1179 
       1180     def evaluate(self,


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/keras/engine/training_arrays.py in fit_loop(model, fit_function, fit_inputs, out_labels, batch_size, epochs, verbose, callbacks, val_function, val_inputs, shuffle, callback_metrics, initial_epoch, steps_per_epoch, validation_steps, validation_freq)
        202                     ins_batch[i] = ins_batch[i].toarray()
        203 
    --> 204                 outs = fit_function(ins_batch)
        205                 outs = to_list(outs)
        206                 for l, o in zip(out_labels, outs):


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py in __call__(self, inputs)
       2983                     'In order to feed symbolic tensors to a Keras model '
       2984                     'in TensorFlow, you need tensorflow 1.8 or higher.')
    -> 2985             return self._legacy_call(inputs)
       2986 
       2987 


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py in _legacy_call(self, inputs)
       2953         session = get_session()
       2954         updated = session.run(fetches=fetches, feed_dict=feed_dict,
    -> 2955                               **self.session_kwargs)
       2956         return updated[:len(self.outputs)]
       2957 


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/client/session.py in run(self, fetches, feed_dict, options, run_metadata)
        893     try:
        894       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 895                          run_metadata_ptr)
        896       if run_metadata:
        897         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/client/session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
       1126     if final_fetches or final_targets or (handle and feed_dict_tensor):
       1127       results = self._do_run(handle, final_targets, final_fetches,
    -> 1128                              feed_dict_tensor, options, run_metadata)
       1129     else:
       1130       results = []


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/client/session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
       1342     if handle is None:
       1343       return self._do_call(_run_fn, self._session, feeds, fetches, targets,
    -> 1344                            options, run_metadata)
       1345     else:
       1346       return self._do_call(_prun_fn, self._session, handle, feeds, fetches)


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
       1348   def _do_call(self, fn, *args):
       1349     try:
    -> 1350       return fn(*args)
       1351     except errors.OpError as e:
       1352       message = compat.as_text(e.message)


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/client/session.py in _run_fn(session, feed_dict, fetch_list, target_list, options, run_metadata)
       1327           return tf_session.TF_Run(session, options,
       1328                                    feed_dict, fetch_list, target_list,
    -> 1329                                    status, run_metadata)
       1330 
       1331     def _prun_fn(session, handle, feed_dict, fetch_list):


    KeyboardInterrupt: 


*Plot*


```python
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

fig, ax = plt.subplots()
plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(bottom=False, left=False)
frame1 = plt.gca()
fig.canvas.draw()
labels = [item.get_text() for item in ax.get_yticklabels()]
labels = list(map(float, labels))
labels = 100*np.array(labels)
labels = list(map(int, labels))
labels = list(map(str, labels))
labels = [label + "%" for label in labels]
ax.set_yticklabels(labels)
plt.savefig("Conv/ConvModelAccuracy.svg", format='svg')

fig, ax = plt.subplots()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(bottom=False, left=False)
frame1 = plt.gca()
# frame1.axes.yaxis.set_ticklabels([])
plt.savefig("Conv/ConvModelLoss.svg", format='svg')
```
