# import openpyxl module
import numpy as np
import openpyxlimport
import xlwt as xlwt
import pandas as pd
from os.path import exists

lags = [1, 2, 3, 4, 5, 6, 7, 8, 9]
neurons = [50, 100, 150]
epochs = [100, 500, 1000]
dropout = [0, 0.2, 0.4]
shuffle = [True, False]
types = ['SimpleStateless', 'StackedStateless', 'Bidirectional']

for rtype in types:
    wb = xlwt.Workbook()
    for l in lags:
        sheet = wb.add_sheet("Lags=" + str(l))
        sheet.write(0, 0, "Neurons:")
        sheet.write(1, 0, "Dropout:")
        sheet.write(2, 0, "Epochs")
        sheet.write(3, 0, "Shuffle")
        col = 1
        for n in neurons:
            for d in dropout:
                for e in epochs:
                    for s in shuffle:
                        filename = "rmse_" + rtype + "_" + str(l) + "_lags_" + str(d) + "_drop_" + str(
                            n) + "_neurons_" + str(e) + "_epochs_" + str(s) + "_shf.csv"
                        if exists(filename):
                            df = pd.read_csv(filename)
                            result = df.describe()
                            sheet.write(0, col, n)
                            sheet.write(1, col, d)
                            sheet.write(2, col, e)
                            sheet.write(3, col, s)
                            sheet.write(4, col, result.index)
                        col = col + 1
    wb.save(rtype +".xlsx")
l=9
m,h,tr,ts,xtr,xts,ytr,yts = experiment('Bidirectional', dat, l, True, None )

prtr = m.predict(xtr)
prtr = np.insert(prtr, 0, np.nan)
prtr = np.insert(prtr, 0, np.nan)
prtr = np.insert(prtr, 0, np.nan)
prtr = np.insert(prtr, 0, np.nan)
prtr = np.insert(prtr, 0, np.nan)
prtr = np.insert(prtr, 0, np.nan)
prtr = np.insert(prtr, 0, np.nan)
prtr = np.insert(prtr, 0, np.nan)
prtr = np.insert(prtr, 0, np.nan)

prts = m.predict(xts)

fig = pyplot.figure(figsize=(12, 6))
fig.suptitle("C. Bidirectional Stateless network")
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax.set_title('Train set')
ax.plot(tr, color='blue', label='Real values')
ax.plot(prtr, color='black', label='Predicted values')
ax.legend()
xval = list(map(int, range(33, 38))) #range(33, 38, int)
ax2.set_title('Test set')
ax2.plot(xval, yts, color='blue', label='Real values')
ax2.plot(xval, prts, color='black', label='Predicted values')
ax2.legend()
pyplot.show()
pyplot.savefig('fig3.eps', format='eps')