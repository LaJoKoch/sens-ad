import os
import pandas as pd
import pickle 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow.keras.models import load_model
from LSTM_forecast import load_mat_data, split_signals

SENSITIV = 1.25 
SEQ_LEN = 100 
model = load_model('models/lstm_current_rotspeed_pos_accelerat')

with open('models/train_Settings.pkl', 'rb') as f:
    num_features, scaler, quantile = pickle.load(f)

data = load_mat_data('data/testSet')
data = data.drop(columns=['position_rad', 'temperature_DegCel'])
inputs, outputs = split_signals(scaler.transform(data), SEQ_LEN)

preds = model.predict(inputs, verbose=1, batch_size=1)
mse_perSignal = ((outputs - preds) ** 2)/num_features
mae = pd.DataFrame(outputs - preds).abs().sum(axis=1).divide(num_features)

prediction = []
for err in mae:
    if err > SENSITIV*quantile:
        #print('ALERT: Anomaly found!')
        prediction.append(1)
    else:
        prediction.append(0)