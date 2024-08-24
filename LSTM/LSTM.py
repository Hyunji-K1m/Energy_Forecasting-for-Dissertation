from properscoring import crps_ensemble
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.layers import Dropout

data = pd.read_csv('/Users/kimhyunji/Desktop/Dissertation/code/data/rea_final_data without seasonal.csv')
data['start_date'] = pd.to_datetime(data['start_date'])
data.set_index('start_date', inplace=True)

print(data)

features = data.columns[:]
print(features)

data_values = data.values
X = data_values[:-1, :]          
target = data_values[1:, :320]   

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaled_X = scaler_X.fit_transform(X)  
scaled_target = scaler_y.fit_transform(target)  

train_size = int(len(scaled_X) * 0.7)
val_size = int(len(scaled_X) * 0.15)
test_size = len(scaled_X) - train_size - val_size

X_train = scaled_X[:train_size]
y_train = scaled_target[:train_size]

X_val = scaled_X[train_size:train_size + val_size]
y_val = scaled_target[train_size:train_size + val_size]

X_test = scaled_X[train_size + val_size:]
y_test = scaled_target[train_size + val_size:]

def create_sequences(X_data, y_data, seq_length, number_of_steps_to_predict=1):
    xs, ys = [], []
    for i in range(len(X_data) - seq_length - number_of_steps_to_predict + 1):
        x = X_data[i:i + seq_length]
        y = y_data[i + seq_length:i + seq_length + number_of_steps_to_predict].flatten()
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


train_seq_length = 120
X_train_seq, y_train_seq = create_sequences(X_train, y_train, train_seq_length, number_of_steps_to_predict=14)


val_test_seq_length = 120
X_val_seq, y_val_seq = create_sequences(X_val, y_val, val_test_seq_length, number_of_steps_to_predict=14)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, val_test_seq_length, number_of_steps_to_predict=14)


print(f"X_train_seq shape: {X_train_seq.shape}")
print(f"y_train_seq shape: {y_train_seq.shape}")
print(f"X_val_seq shape: {X_val_seq.shape}")
print(f"y_val_seq shape: {y_val_seq.shape}")
print(f"X_test_seq shape: {X_test_seq.shape}")
print(f"y_test_seq shape: {y_test_seq.shape}")


model = Sequential()
model.add(LSTM(128, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(320 * 14))  

model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


history = model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=8, validation_data=(X_val_seq, y_val_seq), callbacks=[early_stopping])
y_pred = model.predict(X_test_seq)
y_pred = y_pred.reshape(-1, 14, 320)

def inverse_transform_sequences(scaler, sequences, original_dim):
    sequences_flat = sequences.reshape(-1, original_dim)
    sequences_original = scaler.inverse_transform(sequences_flat)
    sequences_original = sequences_original.reshape(sequences.shape)
    return sequences_original


X_train_seq_original = inverse_transform_sequences(scaler_X, X_train_seq, X_train_seq.shape[-1])
y_train_seq_original = inverse_transform_sequences(scaler_y, y_train_seq, 320)  

X_val_seq_original = inverse_transform_sequences(scaler_X, X_val_seq, X_val_seq.shape[-1])
y_val_seq_original = inverse_transform_sequences(scaler_y, y_val_seq, 320)

X_test_seq_original = inverse_transform_sequences(scaler_X, X_test_seq, X_test_seq.shape[-1])
y_test_seq_original = inverse_transform_sequences(scaler_y, y_test_seq, 320)


print("Original X_train_seq :")
print(X_train_seq_original.shape)
print("\nOriginal y_train_seq:")
print(y_train_seq_original.shape)

print("Original X_test_seq :")
print(X_test_seq_original.shape)
print("\nOriginal y_test_seq:")
print(y_test_seq_original.shape)

y_pred = y_pred.reshape(-1, 14, 320)


y_pred_original = inverse_transform_sequences(scaler_y, y_pred, 320)  
y_test_seq_original = inverse_transform_sequences(scaler_y, y_test_seq, 320) 

#y_test_seq_original_reshaped = y_test_seq_original.reshape(16, 14, 320) -> predict length=30

y_test_seq_original_reshaped = y_test_seq_original.reshape(32, 14, 320)


print(f"y_pred_original shape: {y_pred_original.shape}")
print(f"y_test_seq_original_reshaped shape: {y_test_seq_original_reshaped.shape}")
print(f"y_pred_original shape: {y_pred_original.shape}")
print(f"y_test_seq_original_reshaped shape: {y_test_seq_original_reshaped.shape}")

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))

smape_value = smape(y_test_seq_original_reshaped, y_pred_original)
print(f"sMAPE: {smape_value}")

def mase(y_true, y_pred):
    n = y_true.shape[0]
    d = np.mean(np.abs(np.diff(y_true, axis=0)))
    errors = np.mean(np.abs(y_pred - y_true))
    return errors / d

mase_value = mase(y_test_seq_original_reshaped, y_pred_original)
print(f"MASE: {mase_value}")
def crps(y_true, y_pred):
    crps_values = [crps_ensemble(y_true[i], y_pred[i]) for i in range(len(y_true))]
    return np.mean(crps_values)

crps_value = crps(y_test_seq_original_reshaped, y_pred_original)
print(f"CRPS: {crps_value}")

def crps_sum(y_true, y_pred):
    crps_values = [crps_ensemble(y_true[i], y_pred[i]) for i in range(len(y_true))]
    return np.sum(crps_values)

crps_sum_value = crps_sum(y_test_seq_original_reshaped, y_pred_original)
print(f"CRPS Sum: {crps_sum_value}")

from sklearn.metrics import mean_squared_error
rmse_value = np.sqrt(mean_squared_error(y_test_seq_original_reshaped.reshape(-1, 320), 
                                        y_pred_original.reshape(-1, 320)))
print(f"RMSE: {rmse_value}")

def relative_rmse(y_true, y_pred):
    rmse_value = np.sqrt(mean_squared_error(y_true.reshape(-1, 320), y_pred.reshape(-1, 320)))
    return rmse_value / np.mean(np.abs(y_true))

relative_rmse_value = relative_rmse(y_test_seq_original_reshaped, y_pred_original)
print(f"Relative RMSE: {relative_rmse_value}")
