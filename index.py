from pyspark.ml.regression import LinearRegression
from pyspark.sql.DataFrameReader import spark
import xlrd
import csv


# convert xls file to csv format
wb = xlrd.open_workbook('kauai.xls')
sh = wb.sheet_by_index(0)
kauai_csv = open('kauai_jobs_data.csv', 'w')
wr = csv.writer(kauai_csv, quoting=csv.QUOTE_ALL)
lHeaders = ['year','month','Arts_Entertainment_Recreation','Accommodation','FoodServices_DrinkingPlaces']
wr.writerow(lHeaders)
for rownum in range(6, sh.nrows-7):
    lDataRow = [sh.row_values(rownum)[0],sh.row_values(rownum)[1],sh.row_values(rownum)[20],sh.row_values(rownum)[21],sh.row_values(rownum)[22]]
    wr.writerow(lDataRow)
kauai_csv.close(),


# Load training data
training = spark.read.format("csv")\
    .options(header='true', inferschema='true')\
    .load("Visitor Arrival_State of Hawaii-Monthly.csv")

# lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# # Fit the model
# lrModel = lr.fit(training)

# # Print the coefficients and intercept for linear regression
# print("Coefficients: %s" % str(lrModel.coefficients))
# print("Intercept: %s" % str(lrModel.intercept))

# # Summarize the model over the training set and print out some metrics
# trainingSummary = lrModel.summary
# print("numIterations: %d" % trainingSummary.totalIterations)
# print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
# trainingSummary.residuals.show()
# print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
# print("r2: %f" % trainingSummary.r2)