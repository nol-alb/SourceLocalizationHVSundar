# SAMPLE CNN ARCHITECTURE IMPLEMENTED FOR MUSIC AUTO TAGGING

## Importing Libraries


```python
import numpy as np
import tensorflow as tf

from keras.models import load_model
import wave
import os
import glob
import librosa
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import (Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Multiply, GlobalMaxPool1D,
                                     Dense, Dropout, Activation, Reshape, Concatenate, Add)
from keras.layers import Input
from keras.utils import np_utils
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import pandas as pd
from tensorflow.keras import Model
from keras.layers import InputLayer
```

    Using TensorFlow backend.


## SAMPLE CNN ARCHITECTURE AS PROPOSED BY JONGPIL LEE


```python

    kernel_initializer=tf.keras.initializers.GlorotUniform()
    activation=keras.activations.relu
    dropout_rate=0.1
    classifier = keras.Sequential()
    # 59049 X 1
    classifier.add(InputLayer(input_shape=(59049,1)))
    classifier.add(Conv1D(filters=128,kernel_size=3, padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation(activation))
    classifier.add(MaxPool1D(3))

    # 19683 X 128
    classifier.add(Conv1D(filters=128, kernel_size=3, padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation(activation))
    classifier.add(MaxPool1D(3))
    # 6561 X 128
    classifier.add(Conv1D(128, 3, padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation(activation))
    classifier.add(MaxPool1D(3))
    # 2187 X 128
    classifier.add(Conv1D(256, 3, padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation(activation))
    classifier.add(MaxPool1D(3))
    # 729 X 256
    classifier.add(Conv1D(256, 3, padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation(activation))
    classifier.add(MaxPool1D(3))
    # 243 X 256
    classifier.add(Conv1D(256, 3, padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation(activation))
    classifier.add(MaxPool1D(3))
    # 81 X 256
    classifier.add(Conv1D(256, 3, padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation(activation))
    classifier.add(MaxPool1D(3))
    # 27 X 256
    classifier.add(Conv1D(256, 3, padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation(activation))
    classifier.add(MaxPool1D(3))
    # 9 X 256
    classifier.add(Conv1D(256, 3, padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation(activation))
    classifier.add(MaxPool1D(3))
    # 3 X 256
    classifier.add(Conv1D(512, 3, padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation(activation))
    classifier.add( MaxPool1D(3))
    # 1 X 512
    classifier.add(Conv1D(512, 1, padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation(activation))
    # 1 X 512
    classifier.add(Dropout(dropout_rate))
    classifier.add(Flatten())
    classifier.add(Dense(units=50, activation='sigmoid'))
    classifier.summary()

```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv1d_1 (Conv1D)            (None, 59049, 128)        512       
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 59049, 128)        512       
    _________________________________________________________________
    activation_1 (Activation)    (None, 59049, 128)        0         
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 19683, 128)        0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 19683, 128)        49280     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 19683, 128)        512       
    _________________________________________________________________
    activation_2 (Activation)    (None, 19683, 128)        0         
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 6561, 128)         0         
    _________________________________________________________________
    conv1d_3 (Conv1D)            (None, 6561, 128)         49280     
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 6561, 128)         512       
    _________________________________________________________________
    activation_3 (Activation)    (None, 6561, 128)         0         
    _________________________________________________________________
    max_pooling1d_3 (MaxPooling1 (None, 2187, 128)         0         
    _________________________________________________________________
    conv1d_4 (Conv1D)            (None, 2187, 256)         98560     
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 2187, 256)         1024      
    _________________________________________________________________
    activation_4 (Activation)    (None, 2187, 256)         0         
    _________________________________________________________________
    max_pooling1d_4 (MaxPooling1 (None, 729, 256)          0         
    _________________________________________________________________
    conv1d_5 (Conv1D)            (None, 729, 256)          196864    
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 729, 256)          1024      
    _________________________________________________________________
    activation_5 (Activation)    (None, 729, 256)          0         
    _________________________________________________________________
    max_pooling1d_5 (MaxPooling1 (None, 243, 256)          0         
    _________________________________________________________________
    conv1d_6 (Conv1D)            (None, 243, 256)          196864    
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 243, 256)          1024      
    _________________________________________________________________
    activation_6 (Activation)    (None, 243, 256)          0         
    _________________________________________________________________
    max_pooling1d_6 (MaxPooling1 (None, 81, 256)           0         
    _________________________________________________________________
    conv1d_7 (Conv1D)            (None, 81, 256)           196864    
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 81, 256)           1024      
    _________________________________________________________________
    activation_7 (Activation)    (None, 81, 256)           0         
    _________________________________________________________________
    max_pooling1d_7 (MaxPooling1 (None, 27, 256)           0         
    _________________________________________________________________
    conv1d_8 (Conv1D)            (None, 27, 256)           196864    
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 27, 256)           1024      
    _________________________________________________________________
    activation_8 (Activation)    (None, 27, 256)           0         
    _________________________________________________________________
    max_pooling1d_8 (MaxPooling1 (None, 9, 256)            0         
    _________________________________________________________________
    conv1d_9 (Conv1D)            (None, 9, 256)            196864    
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 9, 256)            1024      
    _________________________________________________________________
    activation_9 (Activation)    (None, 9, 256)            0         
    _________________________________________________________________
    max_pooling1d_9 (MaxPooling1 (None, 3, 256)            0         
    _________________________________________________________________
    conv1d_10 (Conv1D)           (None, 3, 512)            393728    
    _________________________________________________________________
    batch_normalization_10 (Batc (None, 3, 512)            2048      
    _________________________________________________________________
    activation_10 (Activation)   (None, 3, 512)            0         
    _________________________________________________________________
    max_pooling1d_10 (MaxPooling (None, 1, 512)            0         
    _________________________________________________________________
    conv1d_11 (Conv1D)           (None, 1, 512)            262656    
    _________________________________________________________________
    batch_normalization_11 (Batc (None, 1, 512)            2048      
    _________________________________________________________________
    activation_11 (Activation)   (None, 1, 512)            0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 1, 512)            0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 512)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 50)                25650     
    =================================================================
    Total params: 1,875,762
    Trainable params: 1,869,874
    Non-trainable params: 5,888
    _________________________________________________________________


## TRAINING

### Reshaping the training data to fit specifications of CONV1D

#### conv1d(batch_size,time_steps,input_dimensions)


```python
from numpy import array
path ="/Users/noelalben/Desktop/VTU_PROJ/music_autotagging/songs_in_order"
X=[]
for filename in glob.glob(os.path.join(path, '*.wav')):
    x, fs = librosa.load(filename)
    temp=x[0:59049]
    y=[]
    y=temp.tolist()
    X.append(y)
len(X)
X=array(X)
type(X)
X[0].shape
X[0]
sample_size=X.shape[0]
time_steps=X.shape[1]
input_dim=1
X_reshaped=X.reshape(sample_size,time_steps,input_dim)
```


```python
X_reshaped.shape
```




    (30, 59049, 1)



## (30, 59049, 1)

### Acquiring outputs to train the model for 30 songs


```python
out=pd.read_csv('/Users/noelalben/Desktop/VTU_PROJ/music_autotagging/soundoutput.csv')
out=out.to_numpy()
```

### Splitting the data into Training data and test data


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, out, test_size=0.2, random_state=42)
```

### Adding a loss function and training the model


```python
classifier.compile(loss='mse', optimizer='adam')
```


```python
classifier.fit(X_train,y_train,epochs=5)
```

    Epoch 1/5
    24/24 [==============================] - 52s 2s/step - loss: 0.2987
    Epoch 2/5
    24/24 [==============================] - 31s 1s/step - loss: 0.2471
    Epoch 3/5
    24/24 [==============================] - 26s 1s/step - loss: 0.2050
    Epoch 4/5
    24/24 [==============================] - 24s 1s/step - loss: 0.1705
    Epoch 5/5
    24/24 [==============================] - 28s 1s/step - loss: 0.1417





    <keras.callbacks.callbacks.History at 0x1a4f7f5f60>



## TESTING THE MODEL

### Loading test audio


```python
song,fs=librosa.load('/Users/noelalben/Desktop/VTU_PROJ/music_autotagging/songs_in_order/35.mp3')
song=song[0:59049]
song1=np.array(song)
song1=song1.reshape(1,59049,1)
song1.shape
plt.plot(song)


```

    /Users/noelalben/opt/anaconda3/envs/TensorFlow/lib/python3.6/site-packages/librosa/core/audio.py:162: UserWarning: PySoundFile failed. Trying audioread instead.
      warnings.warn("PySoundFile failed. Trying audioread instead.")





    [<matplotlib.lines.Line2D at 0x1a4eef0b00>]




    
![png](output_20_2.png)
    



```python
predi=classifier.predict(song1)
predi = np.where(predi>0.5,1,0)
```

### Our Sample CNN's prediction of tags for the loaded audio


```python
predi
```




    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0]])



## EVALUATING OUR PREDICTION

### Original Dataset audio tags


```python
  df = pd.read_csv("/Users/noelalben/Desktop/VTU_PROJ/music_autotagging/annotations_final.csv", delimiter='\t')

  # Calculate TOP 50 tags.
  top50 = (df.drop(['clip_id', 'mp3_path'], axis=1)
    .sum()
    .sort_values()
    .tail(50)
    .index
    .tolist())

  # Select TOP 50 columns.
  df = df[top50]
  y=df
  y
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
      <th>choral</th>
      <th>female voice</th>
      <th>metal</th>
      <th>country</th>
      <th>weird</th>
      <th>no voice</th>
      <th>cello</th>
      <th>harp</th>
      <th>beats</th>
      <th>female vocal</th>
      <th>...</th>
      <th>piano</th>
      <th>fast</th>
      <th>rock</th>
      <th>electronic</th>
      <th>drums</th>
      <th>strings</th>
      <th>techno</th>
      <th>slow</th>
      <th>classical</th>
      <th>guitar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25858</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25859</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25860</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25861</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25862</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>25863 rows × 50 columns</p>
</div>



### Extracting 35th song tags


```python
z=y[35:36]
z.to_numpy()

```




    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0]])



### Comparing model prediction with Dataset Tags


```python
comparison = predi == z 
equal_arrays = comparison.all() 
  
print(equal_arrays)
```

    choral          True
    female voice    True
    metal           True
    country         True
    weird           True
    no voice        True
    cello           True
    harp            True
    beats           True
    female vocal    True
    male voice      True
    dance           True
    new age         True
    voice           True
    choir           True
    classic         True
    man             True
    solo            True
    sitar           True
    soft            True
    pop             True
    no vocal        True
    male vocal      True
    woman           True
    flute           True
    quiet           True
    loud            True
    harpsichord     True
    no vocals       True
    vocals          True
    singing         True
    male            True
    opera           True
    indian          True
    female          True
    synth           True
    vocal           True
    violin          True
    beat            True
    ambient         True
    piano           True
    fast            True
    rock            True
    electronic      True
    drums           True
    strings         True
    techno          True
    slow            True
    classical       True
    guitar          True
    dtype: bool

