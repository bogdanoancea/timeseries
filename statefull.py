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


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def buildLSTModel(type_, n_neurons, dropout, batch, n_out):
    model = None
    if type_ == 'SimpleStateful':
        model = Sequential()
        model.add(
            LSTM(n_neurons, activation='relu', batch_input_shape=batch, recurrent_dropout=dropout,
                 stateful=True))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

    # if type_ == 'StackedStateless':
    #     model = Sequential()
    #     model.add(LSTM(n_neurons, activation='relu', return_sequences=True, input_shape=(n_lags, n_features),
    #                    recurrent_dropout=dropout, stateful=False))
    #     model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, stateful=False))
    #     model.add(Dense(1))
    # if type_ == 'Bidirectional':
    #     model = Sequential()
    #     model.add(Bidirectional(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, stateful=False),
    #                             input_shape=(n_lags, n_features)))
    #     model.add(Dense(1))
    # if type_ == 'Vector':
    #     model = Sequential()
    #     model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, return_sequences=True,
    #                    input_shape=(n_lags, n_features)))
    #     model.add(LSTM(n_neurons, recurrent_dropout=dropout, activation='relu'))
    #     model.add(Dense(n_out))
    # if type_ == 'Encoder-Decoder':
    #     model = Sequential()
    #     model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, input_shape=(n_lags, n_features)))
    #     model.add(RepeatVector(n_out))
    #     model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, return_sequences=True))
    #     model.add(TimeDistributed(Dense(1)))
    # if model is not None:
    #     model.compile(optimizer='adam', loss='mse')
    return model


# run a repeated experiment
def experiment(type_, df_, lg, shf, nout):
    drop = [0]
    neurons = [150]
    reps = 30
    epchs = [1000]
    # define input sequence
    raw_seq = df_.iloc[:, 1].values
    # raw_seq = raw_seq.to_numpy()
    # raw_diff = difference(raw_seq, 1)
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
                                          batch=(batch_size, X_train.shape[1], X_train.shape[2]), n_out=None)
                    for i in range(e):
                        model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0, shuffle=shf)
                        model.reset_states()

                    yhat = model.predict(X_test, batch_size=batch_size, verbose=0)
                    print(yhat)
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
    results = pd.DataFrame()
    # run experiment
    l = 3
    experiment('SimpleStateful', dat, l, False, None)
