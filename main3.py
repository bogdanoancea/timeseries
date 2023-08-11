from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
from numpy import concatenate

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(lag,0, -1)]
	columns.append(df)
	df = concat(columns, axis=1)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

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
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False, workers = 4, use_multiprocessing= True)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# run a repeated experiment

def experiment(series, n_lag, n_repeats, n_epochs, n_batch, n_neurons):
	# transform data to be stationary
	raw_values = series.values
	diff_values = difference(raw_values, 1)
	# transform data to be supervised learning
	supervised = timeseries_to_supervised(diff_values, n_lag)
	##### !!!!! Aici trebuie sa modific daca modific n_lag: supervised_values = supervised.values[n_lag:,:]
	supervised_values = supervised.values[1:,:]
	# split data into train and test-sets
	train, test = supervised_values[0:-12, :], supervised_values[-12:, :]
	# transform the scale of the data
	scaler, train_scaled, test_scaled = scale(train, test)
	#train_scaled, test_scaled = train.astype(float), test.astype(float)
	# run experiment
	error_scores = list()
	avg_predictions = numpy.full( shape= len(test_scaled), fill_value=0.0)
	for r in range(n_repeats):
		# fit the base model
		lstm_model = fit_lstm(train_scaled, n_batch, n_epochs, n_neurons)
		# forecast test dataset
		predictions = list()
		for i in range(len(test_scaled)):
			# predict
			X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
			yhat = forecast_lstm(lstm_model, 1, X)
			# invert scaling
			yhat = invert_scale(scaler, X, yhat)
			# invert differencing
			yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
			# store forecast
			predictions.append(yhat)
		# report performance
		avg_predictions = avg_predictions + numpy.array(predictions)
		rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
		print('%d) Test RMSE: %.3f' % (r+1, rmse))
		error_scores.append(rmse)
	avg_predictions = avg_predictions / n_repeats
	return error_scores, avg_predictions

# execute the experiment
def run():
	# load dataset
	#series = read_csv('shampoo.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
	series = read_csv('seria.csv', header=0, index_col=0)
	# experiment
	n_lag = 1
	n_repeats = 30
	n_epochs = 1500
	n_batch = 1
	n_neurons = [1,5,10,50]
	# run the experiment
	for n in n_neurons:
		print('No neurons:', n)
		results = DataFrame()
		results['results'], p = experiment(series, n_lag, n_repeats, n_epochs, n_batch, n)
		# summarize results
		print(results.describe())
		# save boxplot
		results.boxplot()
		pyplot.savefig('experiment_stateful'+n+'_neurons.png')
		# save results
		results.to_csv('experiment_stateful'+n+'_neurons.csv', index=False)
		p.to_csv('experiment_stateful_predictions'+n+'_neurons.csv', index=False)
 # entry point
run()


n_batch = 2
X=train_scaled[:, 0:-1]
y = train_scaled[:,-1]
X = X.reshape(X.shape[0]/2, 2, X.shape[1])
y = y.reshape(y.shape[0]/2,2)