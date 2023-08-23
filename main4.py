from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import ConvLSTM2D
from matplotlib import pyplot
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
from math import sqrt
import tensorflow as tf
import statistics as st
import gc
from multiprocessing import Process
from sklearn.preprocessing import MinMaxScaler
# split a univariate sequence into samples
def data_to_supervised(data, n_lags):
    X, y = list(), list()
    for i in range(len(data)):
        # find the end of this pattern
        end_ix = i + n_lags
        # check if we are beyond the sequence
        if end_ix > len(data)-1:
            break
        # gather input and output parts of the pattern
        xs, ys = data[i:end_ix], data[end_ix]
        X.append(xs)
        y.append(ys)
    return np.array(X), np.array(y)


def data_to_supervised_nsteps_out(data, n_lags, n_out):
    X, y = list(), list()
    for i in range(len(data)):
        end_ix = i + n_lags
        out_end_ix = end_ix + n_out
        # check if we are beyond the sequence
        if out_end_ix > len(data):
            break
        xs, ys = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(xs)
        y.append(ys)
    return np.array(X), np.array(y)


def buildLSTMmodel(type_, n_neurons, dropout, n_lags, n_seq, n_features) :
    model = None
    if type_ == 'SimpleStateless' :
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', input_shape=(n_lags, n_features), recurrent_dropout=dropout, stateful=False))
    if type_ == 'StackedStateless':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', return_sequences=True, input_shape=(n_lags, n_features), recurrent_dropout=dropout, stateful=False))
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, stateful=False))
    if type_ == 'Bidirectional':
        model = Sequential()
        model.add(Bidirectional(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, stateful=False), input_shape=(n_lags, n_features)))
    if type_ == 'CNN-LSTM':
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),input_shape=(None, n_lags, n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout))
    if type_ == 'ConvLSTM':
        model = Sequential()
        model.add(
            ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu', recurrent_dropout=dropout,
                       input_shape=(n_seq, 1, n_lags, n_features)))
        model.add(Flatten())

    if model != None:
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
    return model


def buildLSTModel2(type, dropout, n_steps_in, n_features, n_steps_out):
    model = None
    if type == 'Vector' :
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(Dropout(d))
        model.add(LSTM(100, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(n_steps_out))
    if type == 'Encode-Decoder':
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(RepeatVector(n_steps_out))
        model.add(Dropout(dropout))
        model.add(LSTM(100, activation='relu', return_sequences=True))
        model.add(Dropout(dropout))
        model.add(TimeDistributed(Dense(1)))
    if model != None:
        model.compile(optimizer='adam', loss='mse')
    return model

# # scale train and test data to [-1, 1]
# def scale(train, test):
# 	# fit scaler
# 	scaler = MinMaxScaler(feature_range=(-1, 1))
# 	scaler = scaler.fit(train)
# 	# transform train
# 	train = train.reshape(train.shape[0], train.shape[1])
# 	train_scaled = scaler.transform(train)
# 	# transform test
# 	test = test.reshape(test.shape[0], test.shape[1])
# 	test_scaled = scaler.transform(test)
# 	return scaler, train_scaled, test_scaled
#
# # inverse scaling for a forecasted value
# def invert_scale(scaler, X, yhat):
# 	new_row = [x for x in X] + [yhat]
# 	array = numpy.array(new_row)
# 	array = array.reshape(1, len(array))
# 	inverted = scaler.inverse_transform(array)
# 	return inverted[0, -1]

def experiment(type_, df_, l, shf):
    drop = [0, 0.2, 0.4]
    neurons = [50, 100, 150]
    reps = 30
    epchs = [100, 500, 1000]
    # define input sequence
    raw_seq = df_.iloc[range(len(df_.index)), 1]
    raw_seq = raw_seq.to_numpy()
    #scaler = MinMaxScaler(feature_range=(0,1))
    #raw_seq =raw_seq.reshape(-1,1)
    #raw_seq = scaler.fit_transform(raw_seq)
    train = raw_seq[0:-14]
    test = raw_seq[-14:]
    features = 1
    minerr = 1000000
    n_min = None
    d_min = None
    e_min = None
    l_min = None

    X_train, y_train = data_to_supervised(train, l)
    X_test, y_test = data_to_supervised(test, l)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], features))
    for n in neurons:
        #print("Neurons:", n)
        for d in drop:
            #print("drop:", d)
            for e in epchs:
                #print("Epochs:", e)
                error_scores = list()
                avg_predictions = np.full(shape=(len(y_test),1), fill_value=0.0)
                for r in range(reps):
                    model = buildLSTMmodel(type_, n_neurons=n, dropout=d, n_lags=l, n_seq=None, n_features=features)
                    model.fit(X_train, y_train, epochs=e, verbose=0, shuffle=shf)
#                            with open('history_'+ type_ +'_' +str(l) + '_lags_'+ str(d) + "_drop_" + str(n) + "_neurons_" + str(e) + "_epochs_" + str(v) + "_vals_" + str(r) + "_reps", "wb") as history_file:
#                                pickle.dump(history, history_file)
                    yhat = model.predict(X_test, verbose = 0)
                    rmse = sqrt(mean_squared_error(yhat, y_test))
                    avg_predictions = avg_predictions + np.array(yhat)
                    #print('%d) Test RMSE: %.3f' % (r + 1, rmse))
                    error_scores.append(rmse)
                avg_predictions = avg_predictions / reps
                df = pd.DataFrame(avg_predictions)
                df.to_csv('predictions_'+ type_ + '_' + str(l) + '_lags_'+ str(d) + "_drop_" + str(n) + "_neurons_" + str(e) + "_epochs_" + str(shf) +"_shf" +'.csv', index=False, header=False)
                err = pd.DataFrame(error_scores)
                err.to_csv('rmse_' +type_ + "_"+str(l) + '_lags_'+ str(d) + "_drop_" + str(n) + "_neurons_" + str(e) + "_epochs_" + str(shf) +"_shf" +'.csv', index=False, header=False)
                print('Avg. Test RMSE: %.3f l = %d n = %d d = %f e = %d  shf = %d' % (st.mean(error_scores), l, n, d, e, shf))
                mean_err = st.mean(error_scores)
                if mean_err < minerr:
                    minerr = mean_err
                    n_min = n
                    d_min = d
                    e_min = e
                    l_min = l

    print('Minimum Test RMSE: %.3f %d %d %f %d' % (minerr, l_min, n_min, d_min, e_min))



if __name__ == '__main__':
    dat = pd.read_excel('seria.xlsx', header=None)
    print(dat)
    lags=[1,2,3,4,5,6,7,8,9]
    psimple = [0] * 9
    i = 0
    # pstacked = [0] * 9
    # j = 0

    for l in lags:
        psimple[i] = Process(target=experiment, args = ('Bidirectional', dat, l, False))
        psimple[i].start()
        i = i +1
        # pstacked[j] = Process(target=experiment, args = ('StackedStateless', dat, l, False))
        # pstacked[j].start()
        # j = j + 1
    for p in psimple:
        p.join()
    # for p in pstacked:
    #     p.join()

    #experiment('SimpleStateless', dat, lags, shf=False)
    #experiment('StackedStateless', dat, lags, shf=False)
    #experiment('Bidirectional', dat, lags, shf=False)

# model = Sequential()
# model.add(LSTM(50,
#                batch_input_shape=(batch_size, tsteps, 1),
#                return_sequences=True,
#                stateful=True))
# model.add(LSTM(50,
#                return_sequences=False,
#                stateful=True))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='rmsprop')
#
# print('Training')
# for i in range(epochs):
#     print('Epoch', i, '/', epochs)
#     model.fit(cos,
#               expected_output,
#               batch_size=batch_size,
#               verbose=1,
#               nb_epoch=1,
#               shuffle=False)
#     model.reset_states()
#
# print('Predicting')
# predicted_output = model.predict(cos, batch_size=batch_size)
#
# print('Plotting Results')
# plt.subplot(2, 1, 1)
# plt.plot(expected_output)
# plt.title('Expected')
# plt.subplot(2, 1, 2)
# plt.plot(predicted_output)
# plt.title('Predicted')
# plt.show()