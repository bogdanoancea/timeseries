import numpy as np
import openpyxl
import xlwt as xlwt
import pandas as pd
from os.path import exists

lags = [1, 2, 3, 4, 5, 6, 7, 8, 9]
neurons = [50, 100, 150]
epochs = [100, 500, 1000]
dropout = [0, 0.2, 0.4]
shuffle = [True, False]
types = ['SimpleStateless', 'StackedStateless', 'Bidirectional']
types = ['Vector', 'Encoder-Decoder']
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


l=8
m,h,tr,ts,xtr,xts,ytr,yts = experiment('Encoder-Decoder', dat, l, False, 3)

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







fig = pyplot.figure(figsize=(12, 6))
fig.suptitle("Test set")

ax = fig.add_subplot(221)
xval = list(map(int, range(24, 38)))
ax.plot(xval, ts, color='blue', label='Real values')
prts = m.predict(xts)
p0 = prts[0]
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 11, np.nan)
p0 = np.insert(p0, 12, np.nan)
p0 = np.insert(p0, 13, np.nan)
ax.plot(xval, p0, color='red', label='Predicted values')

ax2 = fig.add_subplot(222)
xval = list(map(int, range(24, 38)))
ax2.plot(xval, ts, color='blue', label='Real values')
p1 = prts[1]
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 12, np.nan)
p1 = np.insert(p1, 13, np.nan)
#p1 = np.insert(p1, 13, np.nan)
ax2.plot(xval, p1, color='red', label='Predicted values')

ax3 = fig.add_subplot(223)
xval = list(map(int, range(24, 38)))
ax3.plot(xval, ts, color='blue', label='Real values')
p2 = prts[2]
p2 = np.insert(p2, 0, np.nan)
p2 = np.insert(p2, 0, np.nan)
p2 = np.insert(p2, 0, np.nan)
p2 = np.insert(p2, 0, np.nan)
p2 = np.insert(p2, 0, np.nan)
p2 = np.insert(p2, 0, np.nan)
p2 = np.insert(p2, 0, np.nan)
p2 = np.insert(p2, 0, np.nan)
p2 = np.insert(p2, 0, np.nan)
p2 = np.insert(p2, 0, np.nan)
p2 = np.insert(p2, 13, np.nan)
ax3.plot(xval, p2, color='red', label='Predicted values')

ax4 = fig.add_subplot(224)
xval = list(map(int, range(24, 38)))
ax4.plot(xval, ts, color='blue', label='Real values')
p3 = prts[3]
p3 = np.insert(p3, 0, np.nan)
p3 = np.insert(p3, 0, np.nan)
p3 = np.insert(p3, 0, np.nan)
p3 = np.insert(p3, 0, np.nan)
p3 = np.insert(p3, 0, np.nan)
p3 = np.insert(p3, 0, np.nan)
p3 = np.insert(p3, 0, np.nan)
p3 = np.insert(p3, 0, np.nan)
p3 = np.insert(p3, 0, np.nan)
p3 = np.insert(p3, 0, np.nan)
p3 = np.insert(p3, 0, np.nan)
ax4.plot(xval, p3, color='red', label='Predicted values')
Line, Label = ax4.get_legend_handles_labels()
fig.legend(Line, Label, loc='upper right')
pyplot.show()
pyplot.savefig('fig4.eps', format='eps')