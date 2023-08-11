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

# split a univariate sequence into samples
def data_to_supervised(data, n_lags):
    X, y = list(), list()
    for i in range(len(data)):
        # find the end of this pattern
        end_ix = i + n_lags
        # check if we are beyond the sequence
        if end_ix > len(data):
            break
        # gather input and output parts of the pattern
        xs, ys = data[i:end_ix], data[end_ix - 1:end_ix]
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


def buildLSTMmodel(type, n_neurons, dropout, n_lags, n_seq, n_features) :
    model = None
    if type == 'SimpleStateless' :
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', input_shape=(n_lags, n_features), recurrent_dropout=dropout))
    if type == 'StackedStateless':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', return_sequences=True, input_shape=(n_lags, n_features), recurrent_dropout=dropout))
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout))
    if type == 'Bidirectional':
        model = Sequential()
        model.add(Bidirectional(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout), input_shape=(n_lags, n_features)))
    if type == 'CNN-LSTM':
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),input_shape=(None, n_lags, n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout))
    if type == 'ConvLSTM':
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

def experiment(type, df):
    #params

    lags = [2, 3, 4, 5, 6, 7, 8, 9]
    drop = [0.2,0.4,0.6,0.8]
    neurons = [50, 100, 150, 200, 250, 300, 350]
    reps = 30
    epchs = [500, 1000, 1500, 2000]
    validation = [0.2, 0.4, 0.6]
    # define input sequence
    raw_seq = df.iloc[range(len(df.index)), 1]
    raw_seq = raw_seq.to_numpy()
    train = raw_seq[0:-12]
    test = raw_seq[-12:]
    features = 1
    minerr = 1000000
    n_min = None
    d_min = None
    e_min = None
    v_min = None
    for l in lags:
        print("Lag:", l)
        X_train, y_train = data_to_supervised(train, l)
        X_test, y_test = data_to_supervised(test, l)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], features))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], features))
        for n in neurons:
            print("Neurons:", n)
            for d in drop:
                print("drop:", d)
                for e in epchs:
                    print("Epochs:", e)
                    for v in validation:
                        print("Validation set:", v)
                        error_scores = list()
                        avg_predictions = np.full(shape=(len(y_test),1), fill_value=0.0)
                        for r in tf.range(reps):
                            model = buildLSTMmodel(type, n_neurons=n, dropout=d, n_lags=l, n_seq=None, n_features=features)
                            history = model.fit(X_train, y_train, epochs=e, verbose=0, validation_split=0.4)
                            with open('history_'+ str(l) + '_lags_'+ str(d) + "_drop_" + str(n) + "_neurons_" + str(e) + "_epochs_" + str(v) + "_vals_" + str(r) + "_reps", "wb") as history_file:
                                pickle.dump(history, history_file)
                            yhat = model.predict(X_test, verbose = 0)
                            rmse = sqrt(mean_squared_error(yhat, y_test))
                            avg_predictions = avg_predictions + np.array(yhat)
                            print('%d) Test RMSE: %.3f' % (r + 1, rmse))
                            error_scores.append(rmse)
                        avg_predictions = avg_predictions / reps
                        df = pd.DataFrame(avg_predictions)
                        df.to_csv('predictions_'+ type + '_' + str(l) + '_lags_'+ str(d) + "_drop_" + str(n) + "_neurons_" + str(e) + "_epochs_" + str(v) + "_vals" +'.csv', index=False, header=False)
                        err = pd.DataFrame(error_scores)
                        err.to_csv('rmse_' + str(l) + '_lags_'+ str(d) + "_drop_" + str(n) + "_neurons_" + str(e) + "_epochs_" + str(v) + "_vals" +'.csv', index=False, header=False)
                        print('Avg. Test RMSE: %.3f' % (st.mean(error_scores) ))
                        mean_err = st.mean(error_scores)
                        if(mean_err < minerr):
                            minerr = mean_err
                            n_min = n
                            d_min = d
                            e_min = e
                            v_min = v

    print('Minimum Test RMSE: %.3f %d %d %d %d' % (minerr, n_min, d_min, e_min,v_min))
    gc.collect()



if __name__ == '__main__':
    dat = pd.read_excel('seria.xlsx', header=None)
    print(dat)
    experiment('SimpleStateless', dat)
    experiment('StackedStateless', dat)


   # #params
    # lags = [2, 3, 4, 5, 6, 7, 8, 9]
    # drop = [0.2,0.4,0.6,0.8]
    # neurons = [50, 100, 150, 200, 250, 300, 350]
    # reps = 30
    # epchs = [500, 1000, 1500, 2000]
    # validation = [0.2, 0.4, 0.6]
    # # define input sequence
    # raw_seq = df.iloc[range(len(df.index)), 1]
    # raw_seq = raw_seq.to_numpy()
    #
    # train = raw_seq[0:-12]
    # test = raw_seq[-12:]
    # features = 1
    #
    # # Simple model
    # for l in lags:
    #     print("Lag:", l)
    #     X_train, y_train = data_to_supervised(train, l)
    #     X_test, y_test = data_to_supervised(test, l)
    #     X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], features))
    #     X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], features))
    #     for n in neurons:
    #         print("Neurons:", n)
    #         for d in drop:
    #             print("drop:", d)
    #             for e in epchs:
    #                 print("Epochs:", e)
    #                 for v in validation:
    #                     print("Validation set:", v)
    #                     error_scores = list()
    #                     avg_predictions = np.full(shape=(len(y_test),1), fill_value=0.0)
    #                     for r in range(reps):
    #                         model = buildLSTMmodel('SimpleStateless', n_neurons=n, dropout=d, n_lags=l, n_seq=None, n_features=features)
    #                         history = model.fit(X_train, y_train, epochs=e, verbose=0, validation_split=0.4)
    #                         with open('history_'+ str(l) + '_lags_'+ str(d) + "_drop_" + str(n) + "_neurons_" + str(e) + "_epochs_" + str(v) + "_vals_" + str(r) + "_reps", "wb") as history_file:
    #                             pickle.dump(history, history_file)
    #                         yhat = model.predict(X_test)
    #                         rmse = sqrt(mean_squared_error(yhat, y_test))
    #                         avg_predictions = avg_predictions + np.array(yhat)
    #                         print('%d) Test RMSE: %.3f' % (r + 1, rmse))
    #                         error_scores.append(rmse)
    #                     avg_predictions = avg_predictions / reps
    #                     df = pd.DataFrame(avg_predictions)
    #                     df.to_csv('simple_stateless_' + str(l) + '_lags_'+ str(d) + "_drop_" + str(n) + "_neurons_" + str(e) + "_epochs_" + str(v) + "_vals" +'.csv', index=False, header=False)
    #                     err = pd.DataFrame(error_scores)
    #                     err.to_csv('rmse_' + str(l) + '_lags_'+ str(d) + "_drop_" + str(n) + "_neurons_" + str(e) + "_epochs_" + str(v) + "_vals" +'.csv', index=False, header=False)





#CNN-LSTM model
    print("CNN-LSTM model")
    df = pd.read_excel('seria.xlsx', header=None)
    print(df)
    drop = [0.2, 0.4, 0.6, 0.8]
    raw_seq = df.iloc[range(len(df.index) - 1), 1]
    n_features = 1
    n_seq = 2
    ytrue = df.iloc[len(df.index)-1, 1]
    for n_steps in range(4, 16, 2):
        X, y = split_sequence(raw_seq, n_steps)
        n_s = n_steps / 2
        X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
        x_input = array(df.iloc[range(len(df.index) - n_steps - 1, len(df.index) - 1), 1])
        x_input = x_input.reshape((1, n_seq, n_steps, n_features))
        for d in drop:
            # define input sequence
            yhat = 0
            s = 0
            sp = 0
            model = buildLSTMmodel('CNN-LSTM', d, n_s, n_features)
            for i in range(1,nrep) :
                # fit model
                model.fit(X, y, epochs=200, verbose=0)
                yhat = model.predict(x_input, verbose=0)
                sp += (yhat - ytrue)**2
                s += yhat
                #print(yhat)
            print('yhat mediu = ', s / nrep, 'y =', ytrue, "mse = ", sp / nrep,
                  "n_steps =", n_steps, "drop = ", d)

# ConvLSTM model
    print("ConvLSTM model")
    df = pd.read_excel('seria.xlsx', header=None)
    print(df)
    drop = [0.2, 0.4, 0.6, 0.8]
    raw_seq = df.iloc[range(len(df.index) - 1), 1]
    n_features = 1
    ytrue = df.iloc[len(df.index)-1, 1]
    n_seq = 2
    for n_steps in range(4, 16, 2):
        X, y = split_sequence(raw_seq, n_steps)
        n_s = n_steps / 2
        X = X.reshape((X.shape[0], n_seq, 1, n_s, n_features))
        x_input = array(df.iloc[range(len(df.index) - n_steps - 1, len(df.index)), 1])
        x_input = x_input.reshape((1, n_seq, 1, n_s, n_features))
        for d in drop:
            # split into samples
            yhat = 0
            s = 0
            sp = 0
            model = buildLSTMmodel('ConvLSTM', d, n_s, n_features)
            for i in range(1,nrep) :
                # fit model
                model.fit(X, y, epochs=200, verbose=0)
                yhat = model.predict(x_input, verbose=0)
                sp += (yhat - ytrue)**2
                s += yhat
                #print(yhat)
            print('yhat mediu = ', s / nrep, 'y =', ytrue, "mse = ", sp / nrep,
                  "n_steps =", n_steps, "drop = ", d)

# n_steps forecasting
# Vector output model
    df = pd.read_excel('seria.xlsx', header=None)
    print(df)
    drop = [0.2, 0.4, 0.6, 0.8]
    n_steps_out = 3
    raw_seq = df.iloc[range(len(df.index) - 1), 1]
    n_features = 1
    ytrue = df.iloc[range(len(df.index)-n_steps_out, len(df.index)) , 1]
    for n_steps_in in range(3, 16):
        X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        x_input = array(df.iloc[range(len(df.index) - n_steps_in - n_steps_out, len(df.index) - n_steps_out), 1])
        x_input = x_input.reshape((1, n_steps_in, n_features))
        for d in drop:
            yhat = 0
            s = 0
            sp = 0
            model = buildLSTMmodel2('Vector', d, n_steps_in, n_features, n_steps_out)
            for i in range(1,nrep) :
                model.fit(X, y, epochs=200, verbose=0)
                yhat = model.predict(x_input, verbose=0)
                sp += (yhat - ytrue)**2
                s += yhat
                #print(yhat)
            print('yhat mediu = ', s / nrep, 'y =', df.iloc[len(df.index)-1, 1], "mse = ", sp / nrep,
                  "n_steps =", n_steps, "drop = ", d)


# Encoder-decoder model
    df = pd.read_excel('seria.xlsx', header=None)
    print(df)
    drop = [0.2, 0.4, 0.6, 0.8]
    n_steps_out = 3
    raw_seq = df.iloc[range(len(df.index) - 1), 1]
    n_features = 1
    ytrue = df.iloc[range(len(df.index)-n_steps_out, len(df.index)) , 1]
    for n_steps_in in range(3, 16):
        X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        y = y.reshape((y.shape[0], y.shape[1], n_features))
        x_input = array(df.iloc[range(len(df.index) - n_steps_in - n_steps_out, len(df.index) - n_steps_out), 1])
        x_input = x_input.reshape((1, n_steps_in, n_features))
        for d in drop:
            yhat = 0
            s = 0
            sp = 0
            model = buildLSTMmodel2('Encoder-Decoder', d, n_steps_in, n_features, n_steps_out)
            for i in range(1,nrep) :
                model.fit(X, y, epochs=200, verbose=0)
                yhat = model.predict(x_input, verbose=0)
                sp += (yhat - ytrue)**2
                s += yhat
                #print(yhat)
            print('yhat mediu = ', s / nrep, 'y =', df.iloc[len(df.index)-1, 1], "mse = ", sp / nrep,
                  "n_steps =", n_steps, "drop = ", d)

###########################
N_train = 1000
from numpy.random import choice
one_indexes = choice(a=N_train, size=[round(N_train / 2)], replace=False)
XX_train = [,]
XX_train[one_indexes, 0] = 1  # very long term memory.

########################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM


# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 25
epochs = 25
# number of elements ahead that are used to make the prediction
lahead = 1


def gen_cosine_amp(amp=100, period=1000, x0=0, xn=50000, step=1, k=0.0001):
    """Generates an absolute cosine time series with the amplitude
    exponentially decreasing

    Arguments:
        amp: amplitude of the cosine function
        period: period of the cosine function
        x0: initial x of the time series
        xn: final x of the time series
        step: step of the time series discretization
        k: exponential rate
    """
    cos = np.zeros(((xn - x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i, 0, 0] = amp * np.cos(2 * np.pi * idx / period)
        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
    return cos


print('Generating Data')
cos = gen_cosine_amp()
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
    expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])

print('Output shape')
print(expected_output.shape)

print('Creating Model')
model = Sequential()
model.add(LSTM(50,
               batch_input_shape=(batch_size, tsteps, 1),
               return_sequences=True,
               stateful=True))
model.add(LSTM(50,
               return_sequences=False,
               stateful=True))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    model.fit(cos,
              expected_output,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False)
    model.reset_states()

print('Predicting')
predicted_output = model.predict(cos, batch_size=batch_size)

print('Plotting Results')
plt.subplot(2, 1, 1)
plt.plot(expected_output)
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(predicted_output)
plt.title('Predicted')
plt.show()