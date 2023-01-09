# Author: Laurin Koch 
# Date: 2021
"""
Script to load and preprocess the training data and afterwards train a forecasting LSTM model. 
"""

import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import pickle
import datetime
import mat73
import glob 
import math

# definition of the hyperparameters 
SEQ_LEN = 100 # every input instance is a sequence of length n
VAL_SPLIT = 0.2 # fraction of training data for evaluation after every epoch
BATCH_SIZE = 50 # number of samples per gradient update 
EPOCHS = 15 # number of iterations over the whole dataset
LR = 0.001 # learning rate 
SENSITIV = 1.25 # sensitivity parameter to calibrate the prediction threshold 
experiment = 'lstm_current_rotspeed_pos_accelerat'


def load_csv_data(path, concat=False):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                all_files.append(os.path.join(root, file))
    df_list = [pd.read_csv(file, delimiter='\t', encoding='utf8', header=[1]) for file in all_files]

    df_list = [df.convert_dtypes() for df in df_list]

    try:
        df_list = [df.stack().str.replace(',','.').unstack() for df in df_list]
    except AttributeError:
        pass

    if concat == True:
        df = pd.concat((df_list), axis=0, ignore_index=True)
    else:
        df = df_list

    return df 


def load_mat_data(path):
    df_list = []

    for k, mat_file in enumerate(glob.glob(os.path.join(path, '*.mat'))):
        data_dict = mat73.loadmat(mat_file)
        for i, key in enumerate(data_dict.keys()):
            array = data_dict[key][1]
            if 'speed' in key: # measurements in rad/sec -> 1/sec
                #print('converting speed [rad/sec] in rotational speed [1/sec]...')
                array = (array / (2*math.pi))
            elif 'temperature' in key: # measurements in K -> °C
                #print('converting temp [K] in [°C]...')
                array = array - 273.15
            df = pd.DataFrame(array[0:], columns=[key])
            df_list.append(df)
    df = pd.concat((df_list), axis=1, ignore_index=False)

    return df


def data_labeling(path, sum_error, quantile, startIdx):

    df = load_mat_data(path)

    anomaly = []
    for i in df.index:
        if i >= SEQ_LEN and i <= startIdx:
            anomaly.append(0) #no anomaly
        elif i > startIdx:
            anomaly.append(1) #anomaly
            
    prediction = []
    score = []
    for err in sum_error:
        if err > SENSITIV*quantile:
            prediction.append(1)
            score.append(err)
        else:
            prediction.append(0)
            score.append(err)
    df_evaluate = pd.DataFrame({'ground truth': anomaly, 'prediction': prediction, 'score': score})

    return df_evaluate


def split_signals(signals, seq_len):
    inputs = []
    outputs = []
    for i in range(len(signals)):
        end_idx = i + seq_len

        if end_idx > len(signals)-1:
            break

        seq_in, seq_out = signals[i:end_idx, :], signals[end_idx, :]
        inputs.append(seq_in)
        outputs.append(seq_out)
    return np.array(inputs), np.array(outputs)


def preprocess(path, record=None, file='csv'):
    # standardization of features (sensor signals) by centering (zero mean) and scaling (unit variance)
    scaler = StandardScaler()

    if file == 'mat' :
        data = load_mat_data(path)
        # only position_m, acceleration_radpersec2, current_A, rotspeed_persec
        data = data.drop(columns=['position_rad', 'temperature_DegCel'])
        scal = scaler.fit(data)
        inputs, outputs = split_signals(scaler.transform(data), SEQ_LEN)
        original_in, original_out = split_signals(scaler.inverse_transform(data), SEQ_LEN)
    elif file == 'csv' and record is not None:
        data = load_csv_data(path, concat=False)
        # only use temp inside, current and rotational speed 
        data = data[record].drop(columns=['M1 in Nm ', 'Temp B  ', 'Zyklen'])
        scal = scaler.fit(data)
        inputs, outputs = split_signals(scaler.transform(data), SEQ_LEN)
        original_in, original_out = split_signals(scaler.inverse_transform(data), SEQ_LEN)
    else: # csv and concat
        data = load_csv_data(path, concat=True)
        # only use temp inside, current and rotational speed 
        data = data.drop(columns=['M1 in Nm ', 'Temp B  ', 'Zyklen'])
        scal = scaler.fit(data)
        inputs, outputs = split_signals(scaler.transform(data), SEQ_LEN)
        original_in, original_out = split_signals(scaler.inverse_transform(data), SEQ_LEN)

    num_features = inputs.shape[2]
    print(f'number of features: {num_features}')

    return inputs, outputs, num_features, original_in, original_out, scal


def main():
    #inputs, outputs, num_features = preprocess('data/trainSet', record=6, file='csv')
    #inputs, outputs, num_features = preprocess('data/trainSet', record=None, file='csv')
    inputs, outputs, num_features, _, _, _ = preprocess('data/trainSet', record=None, file='mat')

    # linear stack of layers
    model = Sequential()

    # input_length=seq_len, input_dim=num_features, output_dim=64
    # input is a 3D tensor [batch_size, timesteps, features]
    # cuDNN implementation: activation=tanh, recurrent_activation=sigmoid
    model.add(LSTM(64, input_shape=(SEQ_LEN, num_features), return_sequences=True))
    # dropout layer for regularization: sets randomly fraction of units to zero (prevent overfitting)
    model.add(Dropout(0.1))

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(128))
    model.add(Dropout(0.1))

    # final dense layer (fully connected) with linear activation function (to get unbounded values for regression problem)
    model.add(Dense(num_features, activation='linear'))

    # configure the model 
    adam = optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.compile(optimizer=adam, loss='mae', metrics=['mse'])
    #model.compile(optimizer='rmsprop', loss='mae', metrics=['mse'])

    # stop training process when the loss did not decrease for 10 consecutive epochs 
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    # save best model weights to continue the training later 
    model_checkpoint = ModelCheckpoint(f'models/{experiment}.h5', monitor='val_loss', save_best_only=True, verbose=0, save_weights_only=True)
    # reduce learning rate when loss did not improve for 5 consecutive epochs 
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0, verbose=0)
    
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # train model for number of epochs
    history = model.fit(inputs, outputs,
                    validation_split=VAL_SPLIT,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr, tensorboard_callback]
                    )

    model.load_weights(f'models/{experiment}.h5', by_name=False)

    model.save(f'models/{experiment}')

    with open(f'models/history_{experiment}.pkl', 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == '__main__':
    main()