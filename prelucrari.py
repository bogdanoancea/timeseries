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

## trainset

fig = pyplot.figure(figsize=(7, 12))
fig.suptitle("Train set")

ax = fig.add_subplot(721)
xval = list(map(int, range(0, 24)))
ax.plot(xval, tr, color='blue', label='Real values')
prtr = m.predict(xtr)
p0 = prtr[0]
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
p0 = np.insert(p0, 0, np.nan)
for i in range(11, 24):
    p0 = np.insert(p0, i, np.nan)

ax.plot(xval, p0, color='red', label='Predicted values')

ax2 = fig.add_subplot(722)
ax2.plot(xval, tr, color='blue', label='Real values')
p1 = prtr[1]
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
p1 = np.insert(p1, 0, np.nan)
for i in range(12, 24):
    p1 = np.insert(p1, i, np.nan)
ax2.plot(xval, p1, color='red', label='Predicted values')

ax3 = fig.add_subplot(723)
ax3.plot(xval, tr, color='blue', label='Real values')
p2 = prtr[2]
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
for i in range(13, 24):
    p2 = np.insert(p2, i, np.nan)
ax3.plot(xval, p2, color='red', label='Predicted values')

ax4 = fig.add_subplot(724)
ax4.plot(xval, tr, color='blue', label='Real values')
p3 = prtr[3]
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
for i in range(14, 24):
    p3 = np.insert(p3, i, np.nan)

ax4.plot(xval, p3, color='red', label='Predicted values')

ax5 = fig.add_subplot(725)
ax5.plot(xval, tr, color='blue', label='Real values')
p4 = prtr[4]
p4 = np.insert(p4, 0, np.nan)
p4 = np.insert(p4, 0, np.nan)
p4 = np.insert(p4, 0, np.nan)
p4 = np.insert(p4, 0, np.nan)
p4 = np.insert(p4, 0, np.nan)
p4 = np.insert(p4, 0, np.nan)
p4 = np.insert(p4, 0, np.nan)
p4 = np.insert(p4, 0, np.nan)
p4 = np.insert(p4, 0, np.nan)
p4 = np.insert(p4, 0, np.nan)
p4 = np.insert(p4, 0, np.nan)
p4 = np.insert(p4, 0, np.nan)
for i in range(15, 24):
    p4 = np.insert(p4, i, np.nan)
ax5.plot(xval, p4, color='red', label='Predicted values')


ax6 = fig.add_subplot(726)
ax6.plot(xval, tr, color='blue', label='Real values')
p5 = prtr[5]
p5 = np.insert(p5, 0, np.nan)
p5 = np.insert(p5, 0, np.nan)
p5 = np.insert(p5, 0, np.nan)
p5 = np.insert(p5, 0, np.nan)
p5 = np.insert(p5, 0, np.nan)
p5 = np.insert(p5, 0, np.nan)
p5 = np.insert(p5, 0, np.nan)
p5 = np.insert(p5, 0, np.nan)
p5 = np.insert(p5, 0, np.nan)
p5 = np.insert(p5, 0, np.nan)
p5 = np.insert(p5, 0, np.nan)
p5 = np.insert(p5, 0, np.nan)
p5 = np.insert(p5, 0, np.nan)
for i in range(16, 24):
    p5 = np.insert(p5, i, np.nan)
ax6.plot(xval, p5, color='red', label='Predicted values')

ax7 = fig.add_subplot(727)
ax7.plot(xval, tr, color='blue', label='Real values')
p6 = prtr[6]
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
p6 = np.insert(p6, 0, np.nan)
for i in range(17, 24):
    p6 = np.insert(p6, i, np.nan)
ax7.plot(xval, p6, color='red', label='Predicted values')


ax8 = fig.add_subplot(728)
ax8.plot(xval, tr, color='blue', label='Real values')
p7 = prtr[7]
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
p7 = np.insert(p7, 0, np.nan)
for i in range(18, 24):
    p7 = np.insert(p7, i, np.nan)
ax8.plot(xval, p7, color='red', label='Predicted values')


ax9 = fig.add_subplot(729)
ax9.plot(xval, tr, color='blue', label='Real values')
p8 = prtr[8]
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
p8 = np.insert(p8, 0, np.nan)
for i in range(19, 24):
    p8 = np.insert(p8, i, np.nan)
ax9.plot(xval, p8, color='red', label='Predicted values')

ax10 = fig.add_subplot(7,2,10)
ax10.plot(xval, tr, color='blue', label='Real values')
p9 = prtr[9]
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
p9 = np.insert(p9, 0, np.nan)
for i in range(20, 24):
    p9 = np.insert(p9, i, np.nan)
ax10.plot(xval, p9, color='red', label='Predicted values')

ax11 = fig.add_subplot(7,2,11)
ax11.plot(xval, tr, color='blue', label='Real values')
p10 = prtr[10]
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
p10 = np.insert(p10, 0, np.nan)
for i in range(21, 24):
    p10 = np.insert(p10, i, np.nan)
ax11.plot(xval, p10, color='red', label='Predicted values')

ax12 = fig.add_subplot(7,2,12)
ax12.plot(xval, tr, color='blue', label='Real values')
p11 = prtr[11]
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
p11 = np.insert(p11, 0, np.nan)
for i in range(22, 24):
    p11 = np.insert(p11, i, np.nan)
ax12.plot(xval, p11, color='red', label='Predicted values')



ax13 = fig.add_subplot(7,2,13)
ax13.plot(xval, tr, color='blue', label='Real values')
p12 = prtr[12]
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
p12 = np.insert(p12, 0, np.nan)
for i in range(23, 24):
    p12 = np.insert(p12, i, np.nan)
ax13.plot(xval, p12, color='red', label='Predicted values')


ax14 = fig.add_subplot(7,2,14)
ax14.plot(xval, tr, color='blue', label='Real values')
p13 = prtr[13]
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
p13 = np.insert(p13, 0, np.nan)
ax14.plot(xval, p13, color='red', label='Predicted values')
Line, Label = ax4.get_legend_handles_labels()
fig.legend(Line, Label, loc='upper right')
pyplot.show()
pyplot.savefig('fig5.eps', format='eps')
