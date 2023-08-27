# import openpyxl module
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
