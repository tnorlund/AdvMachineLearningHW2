

```python
import keras
from keras.layers import Input, Conv2D, Flatten, MaxPool2D, Dropout, Dense, LSTM, GRU, Conv1D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
import matplotlib
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
import glob, os 
from skimage import io, transform
import json
```

# Problem 1

Choose a small (< 3,000) image dataset for classification. Include the link where you have downloaded the pictures from.

---

In order for this to work, you need a kaggle account. With this, you can download the data set here [here](https://www.kaggle.com/ivanfel/honey-bee-pollen).

With this, we can read in the data at the path `/images`. This is where you must place the dataset after you download it.


```python
path="images/"
imlist= glob.glob(os.path.join(path, '*.jpg'))
```

Now, we can read all the images, and shape them correctly. The function below reads all the images and returns the array and label for each corresponding label.


```python
def dataset(file_list,size=(300,180),flattened=False):
    data = []
    for i, file in enumerate(file_list):
        image = io.imread(file)
        image = transform.resize(image, size, mode='constant')
        if flattened:
            image = image.flatten()

        data.append(image)

    labels = [1 if f.split("/")[-1][0] == 'P' else 0 for f in file_list]

    return np.array(data), np.array(labels)
X,Y=dataset(imlist)
```

With the images loaded and labeled, we can look at the shape of the data and target of the model.


```python
print('Data:   ',X.shape)
print('Target: ',Y.shape)
```

Here, we see that we have 714 images that are RGB. 

Now lets look at an example from the dataset.


```python
fig, axes = plt.subplots(1,2)
k=0
plt.sca(axes[0])
plt.imshow(X[k])
plt.title('Has Pollen'.format(k, Y[k]))

k=400
plt.sca(axes[1])
plt.imshow(X[k])
plt.title('No Pollen'.format(k, Y[k]));
```

*Goal*: Classify the bees that have pollen and those that do not.

## 1

Train a model from scratch using what little data you have without any regularization, to set a baseline for what can be achieved.

---

The first step in training the model is splitting the train and validation data.


```python
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=18)

partial_x_train, validation_x_train, partial_y_train, validation_y_train = train_test_split(
    x_train, y_train, test_size=0.15, random_state=18)
```

With this, we can compile a model composed of the convolution layers. 


```python
verbose = 0
# input
input_layer = Input(shape=(300, 180, 3))

# conv layers
conv_layer1   = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
conv_layer1   = MaxPool2D( (2, 2), padding='same')(conv_layer1)

conv_layer2   = Conv2D(64, (3, 3), activation='relu')(conv_layer1)
conv_layer2   = MaxPool2D( (2, 2), padding='same')(conv_layer2)

conv_layer3   = Conv2D(128, (3, 3), activation='relu')(conv_layer2)
conv_layer3   = Conv2D(128, (3, 3), activation='relu')(conv_layer3)
conv_layer3   = MaxPool2D( (2, 2), padding='same')(conv_layer3)

conv_layer4   = Conv2D(256, (3, 3), activation='relu')(conv_layer3)
conv_layer4   = Conv2D(256, (3, 3), activation='relu')(conv_layer4)
conv_layer4   = MaxPool2D( (2, 2), padding='same')(conv_layer4)

# flatten and dense layers
flatten_layer = Flatten()(conv_layer3)
dense_layer   = Dense(512, activation='relu')(flatten_layer)

# output
output_layer  = Dense(1, activation='sigmoid')(dense_layer)

model_1 = Model(input_layer, output_layer)
model_1.compile(
    optimizer=RMSprop(lr=1e-4), 
    loss='binary_crossentropy',
    metrics=['accuracy'])
if verbose==1:
    print(model_1.summary())
```


```python
epochs = 100
batch_size = 15
history_1 = model_1.fit(
    partial_x_train, 
    partial_y_train,
    validation_data=(validation_x_train, validation_y_train),
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=verbose
)
```


```python
model_1.save_weights("part1-1.h5")
with open("history_1.json", "w") as fp:
    json.dump(history_1.history, fp)
```


```python
with open("history_1.json") as fp:
    history = json.load(fp)

```


```python
acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r-', label='Training acc')
plt.legend()
plt.title('Training and Validation Acc')
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r-', label='Training acc')
plt.legend()
plt.title('Training and Validation loss')
plt.show()
```


```python
test_loss, test_acc = model_1.evaluate(x_test, y_test, steps=10)
print('The final test accuracy: ',test_acc)
```

## 2

Use data augmentation to generate more training data from your existing training samples. Also add a Dropout layer to your model, right before the densely connected classifier.


```python
# input
input_layer = Input(shape=(300, 180, 3))

# conv layers
conv_layer1   = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
conv_layer1   = MaxPool2D( (2, 2), padding='same')(conv_layer1)

conv_layer2   = Conv2D(64, (3, 3), activation='relu')(conv_layer1)
conv_layer2   = MaxPool2D( (2, 2), padding='same')(conv_layer2)

conv_layer3   = Conv2D(128, (3, 3), activation='relu')(conv_layer2)
conv_layer3   = Conv2D(128, (3, 3), activation='relu')(conv_layer3)
conv_layer3   = MaxPool2D( (2, 2), padding='same')(conv_layer3)

conv_layer4   = Conv2D(256, (3, 3), activation='relu')(conv_layer3)
conv_layer4   = Conv2D(256, (3, 3), activation='relu')(conv_layer4)
conv_layer4   = MaxPool2D( (2, 2), padding='same')(conv_layer4)

# flatten and dense layers
flatten_layer = Flatten()(conv_layer3)
flatten_layer = Dropout(0.5)(flatten_layer)
dense_layer   = Dense(512, activation='relu')(flatten_layer)

# output
output_layer  = Dense(1, activation='sigmoid')(dense_layer)

model_2 = Model(input_layer, output_layer)
model_2.compile(
    optimizer=RMSprop(lr=1e-4), 
    loss='binary_crossentropy',
    metrics=['accuracy'])
if verbose==1:
    print(model_2.summary())
```


```python
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
datagen.fit(partial_x_train)
history_2 = model_2.fit_generator(
    datagen.flow(
        partial_x_train, 
        partial_y_train, 
        batch_size=batch_size
    ),
    steps_per_epoch=len(partial_x_train) / batch_size,
    epochs=epochs,
    verbose=verbose
)
```


```python
model_2.save_weights("part1-2.h5")
```


```python
acc = history_2.history['acc']
val_acc = history_2.history['val_acc']
loss = history_2.history['loss']
val_loss = history_2.history['val_loss']
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r-', label='Training acc')
plt.legend()
plt.title('Training and Validation Acc')
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r-', label='Training acc')
plt.legend()
plt.title('Training and Validation loss')
plt.show()
```


```python
test_loss, test_acc = model_2.evaluate(x_test, y_test, steps=10)
print('The final test accuracy: ',test_acc)
```

## Problem 2

For this problem you use the data in “admData.csv” on Canvas. This file contains the accumulative number of admitted students to a certain program with 5 annual start dates.

The data has seasonal behavior: the accumulative number of admissions is monotonically increasing during the interval between 2 start dates and then it resets once a new term starts as shown in figure 1.

The objective of the problem is to predict the accumulative number of admissions 7, 14, 21, 28, 35, 42, 49, 56, 63, and 70 days from the current date.

![Figure 1](images/Figure1.png)

i.e. on any given day, you need to forecast what the accumulative number of admissions will be in 1 week, 2 weeks, ... 10 weeks from that day.
You need to use 70% of the data for training, 15% for validation, and 15% (the most recent) for test.

## 1

Create a recurrent neural network model. Explore both GRU and LSTM layers.

---

### Reading the Data

So here, we separate the year, month, and day of the date.


```python
df = pd.read_csv(
    filepath_or_buffer="admData.csv"
)
df["month"] = df['InquiryDate'].astype(str).str.split("/").str[0].astype(int)
df["day"] = df['InquiryDate'].astype(str).str.split("/").str[1].astype(int)
df["year"] = df['InquiryDate'].astype(str).str.split("/").str[2].astype(int)
dates = df['InquiryDate'].tolist()
df['InquiryDate'] = pd.to_datetime(df['InquiryDate'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>InquiryDate</th>
      <th>DailyAdmission</th>
      <th>month</th>
      <th>day</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-08-22</td>
      <td>41</td>
      <td>8</td>
      <td>22</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-08-23</td>
      <td>47</td>
      <td>8</td>
      <td>23</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-08-24</td>
      <td>56</td>
      <td>8</td>
      <td>24</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-08-25</td>
      <td>63</td>
      <td>8</td>
      <td>25</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-08-26</td>
      <td>70</td>
      <td>8</td>
      <td>26</td>
      <td>2017</td>
    </tr>
  </tbody>
</table>
</div>



### Feature Extraction

Based on a number of days to look back, we can add additional columns based on the "Daily Admission" those number of days looking back. We must note that the largest of the number of the days looking back reduces our dataset.


```python
days = [3, 1, 9, 5]

def get_features_by_days(df, days):
    days.sort()
    for num_days_back in days:
        _tmp = [None] * num_days_back
        _tmp.extend(df.iloc[:-num_days_back, 0].values.tolist())
        df["{}DaysBack".format(num_days_back)] = _tmp
    df = df.dropna()
    cols = df.columns.tolist()
    return df[cols[1:] + [cols[0]]]
get_features_by_days(df, days).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month</th>
      <th>day</th>
      <th>year</th>
      <th>1DaysBack</th>
      <th>3DaysBack</th>
      <th>5DaysBack</th>
      <th>9DaysBack</th>
      <th>DailyAdmission</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>8</td>
      <td>31</td>
      <td>2017</td>
      <td>86.0</td>
      <td>73.0</td>
      <td>70.0</td>
      <td>41.0</td>
      <td>92</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9</td>
      <td>1</td>
      <td>2017</td>
      <td>92.0</td>
      <td>77.0</td>
      <td>70.0</td>
      <td>47.0</td>
      <td>94</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9</td>
      <td>2</td>
      <td>2017</td>
      <td>94.0</td>
      <td>86.0</td>
      <td>73.0</td>
      <td>56.0</td>
      <td>94</td>
    </tr>
    <tr>
      <th>12</th>
      <td>9</td>
      <td>3</td>
      <td>2017</td>
      <td>94.0</td>
      <td>92.0</td>
      <td>77.0</td>
      <td>63.0</td>
      <td>94</td>
    </tr>
    <tr>
      <th>13</th>
      <td>9</td>
      <td>4</td>
      <td>2017</td>
      <td>94.0</td>
      <td>94.0</td>
      <td>86.0</td>
      <td>70.0</td>
      <td>94</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's look at the dramatic decreases we have in the training dataset.


```python
peaks = [66, 150, 219, 289]

fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(range(df["DailyAdmission"].shape[0]), df["DailyAdmission"])
plt.scatter(df.iloc[peaks, :].index.tolist(), df.iloc[peaks, :]["DailyAdmission"].tolist(), s=100, c="red", marker="x")
plt.scatter(df.iloc[peaks, :].index.tolist(), df.iloc[peaks, :]["DailyAdmission"].tolist(), s=130, facecolors='none', edgecolors='r')
[plt.text(_labels[0]+5, _labels[1]+5, _labels[2], fontsize=10) for _labels in list(zip(
    df.iloc[peaks, :].index.tolist(), 
    df.iloc[peaks, :]["DailyAdmission"].tolist(), 
    (df.iloc[peaks, :]["month"].astype(str) + "/" + df.iloc[peaks, :]["day"].astype(str) + "/" + df.iloc[peaks, :]["year"].astype(str)).tolist()
))]
fig.canvas.draw()
labels = [item.get_text() for item in ax.get_xticklabels()]
_tmp = [labels[0]]
labels =labels[1:]
_tmp.extend([dates[int(label)] for label in labels if not label[0] == "−" and int(label) < len(dates)])
ax.set_xticklabels(_tmp)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
fig.suptitle('Max Admissions', fontsize=20)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Number of Admissions', fontsize=16)
plt.show()
```


![png](images/output_29_0.png)


So we can see that there are definite peaks in the time series. While the first and last recordings are 8/22/2017 and 8/16/2018 respectively, we can assume that the next peak will be half-way between those dates minus the year: 8/18. In other words, the number of admissions will drastically decrease after these days:
 - 1/20
 - 3/31
 - 6/9
 - 8/18
 - 10/28


```python
df["DaysInYear"] = df['InquiryDate'].map(lambda x: x.month*31 + x.day - 31)
```


```python
[plt.text(_labels[0], _labels[1], _labels[2]) for _labels in list(zip(
    df.iloc[peaks, :].index.tolist(), 
    df.iloc[peaks, :]["DailyAdmission"].tolist(), 
    (df.iloc[peaks, :]["month"].astype(str) + "/" + df.iloc[peaks, :]["day"].astype(str) + "/" + df.iloc[peaks, :]["year"].astype(str)).tolist()
))]
plt.show()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/IPython/core/formatters.py in __call__(self, obj)
        339                 pass
        340             else:
    --> 341                 return printer(obj)
        342             # Finally look for special method names
        343             method = get_real_method(obj, self.print_method)


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/IPython/core/pylabtools.py in <lambda>(fig)
        242 
        243     if 'png' in formats:
    --> 244         png_formatter.for_type(Figure, lambda fig: print_figure(fig, 'png', **kwargs))
        245     if 'retina' in formats or 'png2x' in formats:
        246         png_formatter.for_type(Figure, lambda fig: retina_figure(fig, **kwargs))


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/IPython/core/pylabtools.py in print_figure(fig, fmt, bbox_inches, **kwargs)
        126 
        127     bytes_io = BytesIO()
    --> 128     fig.canvas.print_figure(bytes_io, **kw)
        129     data = bytes_io.getvalue()
        130     if fmt == 'svg':


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/matplotlib/backend_bases.py in print_figure(self, filename, dpi, facecolor, edgecolor, orientation, format, bbox_inches, **kwargs)
       2080                     orientation=orientation,
       2081                     bbox_inches_restore=_bbox_inches_restore,
    -> 2082                     **kwargs)
       2083             finally:
       2084                 if bbox_inches and restore_bbox:


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/matplotlib/backends/backend_agg.py in print_png(self, filename_or_obj, metadata, pil_kwargs, *args, **kwargs)
        525 
        526         else:
    --> 527             FigureCanvasAgg.draw(self)
        528             renderer = self.get_renderer()
        529             with cbook._setattr_cm(renderer, dpi=self.figure.dpi), \


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/matplotlib/backends/backend_agg.py in draw(self)
        384         Draw the figure using the renderer.
        385         """
    --> 386         self.renderer = self.get_renderer(cleared=True)
        387         with RendererAgg.lock:
        388             self.figure.draw(self.renderer)


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/matplotlib/backends/backend_agg.py in get_renderer(self, cleared)
        397                           and getattr(self, "_lastKey", None) == key)
        398         if not reuse_renderer:
    --> 399             self.renderer = RendererAgg(w, h, self.figure.dpi)
        400             self._lastKey = key
        401         elif cleared:


    /opt/homebrew/Cellar/python36/3.6.2+_254.20170915/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/matplotlib/backends/backend_agg.py in __init__(self, width, height, dpi)
         84         self.width = width
         85         self.height = height
    ---> 86         self._renderer = _RendererAgg(int(width), int(height), dpi)
         87         self._filter_renderers = []
         88 


    ValueError: Image size of 96839x91799 pixels is too large. It must be less than 2^16 in each direction.



    <Figure size 432x288 with 1 Axes>



```python

```

### Partitioning the Data

Here, we set 70% of the data for training, 15% for validation, and the last 15% for testing the data.


```python
train_portion = round(df.shape[0] * 0.7)
validation_portion = round(df.shape[0] * 0.15)
train_data = df[["DailyAdmission"]][0:train_portion].values
validation_data = df[["DailyAdmission"]][train_portion:train_portion+validation_portion].values
test_data = df[["DailyAdmission"]][train_portion+validation_portion:].values
```

### Normalizing the Data


```python
sc = MinMaxScaler(feature_range=(0,1))
train_data = train_data.reshape(-1,1)
validation_data = validation_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)
```

Here we use the scale factor, `sc`, for re-purposing the data to fit in a range between 0 and 1.


```python
sc.fit(train_data)
norm_train = sc.transform(train_data)
norm_validation = sc.transform(validation_data)
norm_test = sc.transform(test_data)
```

### Creating Sequences

Here we can use a function to develop sequences to train our model with.


```python
def create_sequence(dataset, look_back=5, foresight=4):
    X, Y = [], []
    for i in range(len(dataset)-look_back-foresight):
        observations = dataset[i:(i+look_back), 0] # Sequence of look back
        X.append(observations)
        Y.append(dataset[i+(look_back+foresight), 0])
    return np.array(X), np.array(Y)
```


```python
norm_train_x, norm_train_y = create_sequence(norm_train)
norm_validation_x, norm_validation_y = create_sequence(norm_validation)
norm_test_x, norm_test_y = create_sequence(norm_test)
norm_train_x = np.reshape(norm_train_x, (norm_train_x.shape[0],norm_train_x.shape[1],1))
norm_validation_x = np.reshape(norm_validation_x, (norm_validation_x.shape[0],norm_validation_x.shape[1],1))
norm_test_x = np.reshape(norm_test_x, (norm_test_x.shape[0],norm_test_x.shape[1],1))
```

### LSTM and GRU Models


```python
def create_lstm_gru_model(
    norm_train_x, 
    filter_size=32, 
    dropout=0.1, 
    recurrent_dropout=0.1, 
    lstm=True
):
    model = Sequential()
    if lstm:
        model.add(
            LSTM(
                filter_size, 
                input_shape=(norm_train_x.shape[1],1), 
                dropout=dropout, 
                recurrent_dropout=recurrent_dropout
            )
        )
    else:
        model.add(
            GRU(
                filter_size, 
                input_shape=(norm_train_x.shape[1],1), 
                dropout=dropout, 
                recurrent_dropout=recurrent_dropout
            )
        )
    model.add(
        Dense(
            1, 
            activation="linear"
        )
    )
    model.compile(
        loss="mae", 
        optimizer="adam",
        metrics=["mean_absolute_error"]
    )
    return model
```


```python
model = create_lstm_gru_model(
    norm_train_x, 
    filter_size=32, 
    dropout=0.9, 
    recurrent_dropout=0.3, 
    lstm=True
)
history = model.fit(
    norm_train_x, 
    norm_train_y,
    validation_data=(
        norm_validation_x,
        norm_validation_y
    ),
    epochs=100,
    batch_size=64,
    callbacks=[],
    verbose=0
)
fig = plt.figure(figsize=(15,7))
fig.suptitle("LSTM Network", fontsize=25)
plt.subplot(1,2,1)
plt.title("Loss", fontsize=15)
plt.plot(history.history["loss"], label="Test Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.subplot(1,2,2)
plt.title("MAE", fontsize=15)
plt.plot(history.history["mean_absolute_error"], label="Mean Absolute Error")
plt.legend()
plt.show()
```


![png](images/output_45_0.png)


### 1D CONV Models


```python
def create_conv_model(
    norm_train_x, 
    filter_size=32, 
    dropout=0.1, 
    recurrent_dropout=0.1,
    kernel_size=3,
    lstm=True
):
    model = Sequential()
    model.add(
        Conv1D(
            filter_size,
            input_shape=(norm_test_x.shape[1],1),
            kernel_size=kernel_size,
            padding="same"
        )
    )
    if lstm:
        model.add(
            LSTM(
                filter_size, 
                input_shape=(norm_train_x.shape[1],1), 
                dropout=dropout, 
                recurrent_dropout=recurrent_dropout
            )
        )
    else:
        model.add(
            GRU(
                filter_size, 
                input_shape=(norm_train_x.shape[1],1), 
                dropout=dropout, 
                recurrent_dropout=recurrent_dropout
            )
        )
    model.add(
        Dense(
            1, 
            activation="linear"
        )
    )
    model.compile(
        loss="mae", 
        optimizer="adam",
        metrics=["mean_absolute_error"]
    )
    return model
```


```python
model = create_conv_model(
    norm_train_x, 
    filter_size=32, 
    dropout=0.1, 
    recurrent_dropout=0.1, 
    lstm=True
)
history = model.fit(
    norm_train_x, 
    norm_train_y,
    validation_data=(
        norm_validation_x,
        norm_validation_y
    ),
    epochs=100,
    batch_size=64,
    callbacks=[],
    verbose=0
)
fig = plt.figure(figsize=(15,7))
fig.suptitle("1D-CONV Network", fontsize=25)
plt.subplot(1,2,1)
plt.title("Loss", fontsize=15)
plt.plot(history.history["loss"], label="Test Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.subplot(1,2,2)
plt.title("MAE", fontsize=15)
plt.plot(history.history["mean_absolute_error"], label="Mean Absolute Error")
plt.legend()
plt.show()
```


![png](output_48_0.png)



```python

```
