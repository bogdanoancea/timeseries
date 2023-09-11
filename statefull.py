import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import numpy as np
import statistics as st


def data_to_supervised(data, n_lags, n_out):
    X, y = list(), list()
    if n_out is None:
        for i in range(len(data)):
            end_ix = i + n_lags
            # check if we are beyond the sequence
            if end_ix > len(data) - 1:
                break
            xs, ys = data[i:end_ix], data[end_ix]
            X.append(xs)
            y.append(ys)
    else:
        for i in range(len(data)):
            end_ix = i + n_lags
            out_end_ix = end_ix + n_out
            if out_end_ix > len(data):
                break
            xs, ys = data[i:end_ix], data[end_ix:out_end_ix]
            X.append(xs)
            y.append(ys)

    return np.array(X), np.array(y)


def buildLSTModel(type_, n_neurons, dropout, batch, n_out):
    model = None
    if type_ == 'SimpleStateful':
        model = Sequential()
        model.add(
            LSTM(n_neurons, activation='relu', batch_input_shape=batch, recurrent_dropout=dropout,
                 stateful=True))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

    if type_ == 'StackedStateful':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', return_sequences=True, batch_input_shape=batch,
                       recurrent_dropout=dropout, stateful=True))
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, stateful=True))
        model.add(Dense(1))
    if type_ == 'BidirectionalStateful':
        model = Sequential()
        model.add(Bidirectional(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, stateful=True),
                                batch_input_shape=batch))
        model.add(Dense(1))
    if type_ == 'VectorStateful':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, return_sequences=True, stateful=True,
                       batch_input_shape=batch))
        model.add(LSTM(n_neurons, recurrent_dropout=dropout, activation='relu', stateful=True))
        model.add(Dense(n_out))
    if type_ == 'Encoder-DecoderStateful':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, batch_input_shape=batch, stateful=True))
        model.add(RepeatVector(n_out))
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, return_sequences=True, stateful=True))
        model.add(TimeDistributed(Dense(1)))
    if model is not None:
        model.compile(optimizer='adam', loss='mse')
    return model


def experiment(type_, df_, lg, shf, nout):
    drop = [0, 0.2, 0.4]
    neurons = [50, 100, 150]
    reps = 30
    epchs = [100, 500, 1000]

    raw_seq = df_.iloc[:, 1].values
    train = raw_seq[0:-14]
    test = raw_seq[-14:]
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

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    batch_size = 1
    for n in neurons:
        # print("Neurons:", n)
        for d in drop:
            # print("drop:", d)
            for e in epchs:
                error_scores = list()
                avg_predictions = np.full(shape=(len(y_test), 1), fill_value=0.0)

                for r in range(reps):

                    model = buildLSTModel(type_, n_neurons=n, dropout=d,
                                          batch=(batch_size, X_train.shape[1], X_train.shape[2]), n_out=nout)
                    for i in range(e):
                        model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0, shuffle=shf)
                        model.reset_states()

                    yhat = model.predict(X_test, batch_size=batch_size, verbose=0)
                    #print(yhat)
                    rmse = sqrt(mean_squared_error(yhat, y_test))
                    avg_predictions = avg_predictions + np.array(yhat)
                    print('%d) Test RMSE: %.3f' % (r + 1, rmse))
                    error_scores.append(rmse)
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
                if mean_err < minerr:
                    minerr = mean_err
                    n_min = n
                    d_min = d
                    e_min = e
                    l_min = lg

    print('Minimum Test RMSE: %.3f %d %d %f %d' % (minerr, l_min, n_min, d_min, e_min))


# execute the experiment
if __name__ == '__main__':
    dat = pd.read_excel('seria.xlsx', header=None)
    repeats = 30
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
            psimple[i] = Process(target=experiment, args=('SimpleStateful', dat, l, s, None))
            psimple[i].start()
            i = i + 1
            pstacked[j] = Process(target=experiment, args=('StackedStateful', dat, l, s, None))
            pstacked[j].start()
            j = j + 1
            pbidir[k] = Process(target=experiment, args=('BidirectionalStateful', dat, l, s, None))
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
            pvector[ii] = Process(target=experiment, args=('VectorStateful', dat, l, s, 3))
            pvector[ii].start()
            ii = ii + 1

            pencoder[jj] = Process(target=experiment, args=('Encoder-DecoderStateful', dat, l, s, 3))
            pencoder[jj].start()
            jj = jj + 1

        for p in pvector:
            p.join()
        for p in pencoder:
            p.join()
