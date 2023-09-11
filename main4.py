from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
import statistics as st
from multiprocessing import Process
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import RepeatVector


def data_to_supervised(data, n_lags, n_out):
    X, y = list(), list()
    if n_out is None:
        for i in range(len(data)):
            # find the end of this pattern
            end_ix = i + n_lags
            # check if we are beyond the sequence
            if end_ix > len(data) - 1:
                break
            # gather input and output parts of the pattern
            xs, ys = data[i:end_ix], data[end_ix]
            X.append(xs)
            y.append(ys)
    else:
        for i in range(len(data)):
            end_ix = i + n_lags
            out_end_ix = end_ix + n_out
            # check if we are beyond the sequence
            if out_end_ix > len(data):
                break
            xs, ys = data[i:end_ix], data[end_ix:out_end_ix]
            X.append(xs)
            y.append(ys)

    return np.array(X), np.array(y)


def buildLSTModel(type_, n_neurons, dropout, n_lags, n_features, n_out):
    model = None
    if type_ == 'SimpleStateless':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', input_shape=(n_lags, n_features), recurrent_dropout=dropout,
                       stateful=False))
        model.add(Dense(1))
    if type_ == 'StackedStateless':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', return_sequences=True, input_shape=(n_lags, n_features),
                       recurrent_dropout=dropout, stateful=False))
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, stateful=False))
        model.add(Dense(1))
    if type_ == 'Bidirectional':
        model = Sequential()
        model.add(Bidirectional(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, stateful=False),
                                input_shape=(n_lags, n_features)))
        model.add(Dense(1))
    if type_ == 'Vector':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, return_sequences=True,
                       input_shape=(n_lags, n_features)))
        model.add(LSTM(n_neurons, recurrent_dropout=dropout, activation='relu'))
        model.add(Dense(n_out))
    if type_ == 'Encoder-Decoder':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, input_shape=(n_lags, n_features)))
        model.add(RepeatVector(n_out))
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
    if model is not None:
        model.compile(optimizer='adam', loss='mse')
    return model


def experiment(type_, df_, lg, shf, nout):
    model = None
    history = None
    drop = [0, 0.2, 0.4]
    neurons = [50, 100, 150]
    reps = 30
    epchs = [100, 500, 1000]

    raw_seq = df_.iloc[range(len(df_.index)), 1]
    raw_seq = raw_seq.to_numpy()
    # scaler = MinMaxScaler(feature_range=(0,1))
    # raw_seq =raw_seq.reshape(-1,1)
    # raw_seq = scaler.fit_transform(raw_seq)
    train = raw_seq[0:-14]
    test = raw_seq[-14:]
    features = 1
    minerr = 1000000
    n_min = None
    d_min = None
    e_min = None
    l_min = None
    if nout is None:
        X_train, y_train = data_to_supervised(train, lg, None)
        X_test, y_test = data_to_supervised(test, lg, None)
    else:
        X_train, y_train = data_to_supervised(train, lg, nout)
        X_test, y_test = data_to_supervised(test, lg, nout)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], features))
    for n in neurons:
        # print("Neurons:", n)
        for d in drop:
            # print("drop:", d)
            for e in epchs:
                # print("Epochs:", e)
                error_scores = list()
                #mape_scores = list()
                avg_predictions = np.full(shape=(len(y_test), 1), fill_value=0.0)
                for r in range(reps):
                    model = buildLSTModel(type_, n_neurons=n, dropout=d, n_lags=lg, n_features=features, n_out=nout)
                    history = model.fit(X_train, y_train, epochs=e, verbose=0, shuffle=shf)
                    yhat = model.predict(X_test, verbose=0)
                    if type_ == 'Encoder-Decoder':
                        yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
                    rmse = sqrt(mean_squared_error(yhat, y_test))
                    # mape = mean_absolute_percentage_error(y_test, yhat)
                    # print("MAPE:", mape)
                    avg_predictions = avg_predictions + np.array(yhat)
                    print('%d) Test RMSE: %.3f' % (r + 1, rmse))
                    error_scores.append(rmse)
                    # mape_scores.append(mape)
                avg_predictions = avg_predictions / reps
                df = pd.DataFrame(avg_predictions)
                df.to_csv(
                    'predictions_' + type_ + '_' + str(lg) + '_lags_' + str(d) + "_drop_" + str(n) + "_neurons_" + str(
                        e) + "_epochs_" + str(shf) + "_shf" + '.csv', index=False, header=False)
                err = pd.DataFrame(error_scores)
                err.to_csv('rmse_' + type_ + "_" + str(lg) + '_lags_' + str(d) + "_drop_" + str(n) + "_neurons_" + str(
                    e) + "_epochs_" + str(shf) + "_shf" + '.csv', index=False, header=False)
                print('Avg. Test RMSE: %.3f l = %d n = %d d = %f e = %d  shf = %d' % (
                st.mean(error_scores), lg, n, d, e, shf))
                mean_err = st.mean(error_scores)
                # mean_mape = st.mean(mape_scores)
                if mean_err < minerr:
                    minerr = mean_err
                    n_min = n
                    d_min = d
                    e_min = e
                    l_min = lg

    print('Minimum Test RMSE: %.3f %d %d %f %d' % (minerr, l_min, n_min, d_min, e_min))
    # print("AVG mape: %.3f", mean_mape)
    return model, history, train, test, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    dat = pd.read_excel('seria.xlsx', header=None)
    print(dat)
    shf = [True, False]
    lags = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for s in shf:
        psimple = [0] * 9
        i = 0
        pstacked = [0] * 9
        j = 0
        pbidir = [0] * 9
        k = 0
        for l in lags:
            psimple[i] = Process(target=experiment, args=('SimpleStateless', dat, l, s, None))
            psimple[i].start()
            i = i + 1
            pstacked[j] = Process(target=experiment, args=('StackedStateless', dat, l, s, None))
            pstacked[j].start()
            j = j + 1
            pbidir[k] = Process(target=experiment, args=('Bidirectional', dat, l, s, None))
            pbidir[k].start()
            k = k + 1

        for p in psimple:
            p.join()
        for p in pstacked:
            p.join()
        for p in pbidir:
            p.join()

    lags = [4, 5, 6, 7, 8, 9]
    for s in shf:
        pvector = [0] * 6
        ii = 0
        pencoder = [0] * 6
        jj = 0

        for l in lags:
            pvector[ii] = Process(target=experiment, args=('Vector', dat, l, s, 3))
            pvector[ii].start()
            ii = ii + 1

            pencoder[jj] = Process(target=experiment, args=('Encoder-Decoder', dat, l, s, 3))
            pencoder[jj].start()
            jj = jj + 1

        for p in pvector:
            p.join()
        for p in pencoder:
            p.join()

