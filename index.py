# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import xlrd
import csv
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)
import pandas

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

# Load visitor data into pandas dataframe for unipoviting later.  Just keep the
# header row, total visitors, domestic, and international rows.
csv_visitors_pandadf = pandas.read_csv('Visitor Arrival_State of Hawaii-Monthly.csv', nrows=3)
# Load job data
dfJobs = spark.read.format("csv")\
    .options(header='true', inferschema='true')\
    .load("kauai_jobs_data.csv")

# Unpivot visitor data so years and moths are in nrows
csv_visitors_pandadf = csv_visitors_pandadf.unstack(level=1).reset_index(name='visitor_count')
# in column level_1 after pivot, 0 = Total Visitor Arrivals, 1 = Domestic
# Visitor Arrivals, 2 = International Visitor Arrivals
# pandas uses NaN for null.  Use .dropna() to drop rows with no visitor count
csv_visitors_clean = csv_visitors_pandadf.loc[ (csv_visitors_pandadf['level_0'] != 'Series') ].dropna()
csv_visitors_clean = csv_visitors_clean.rename(columns={'level_0': 'year-month', 'level_1': 'visitor_type_id'})
# convert pandas dataframe to spark
dfVisitors = spark.createDataFrame(csv_visitors_clean)
dfMapMonths = spark.read.format("csv")\
    .options(header='true', inferschema='true')\
    .load('map_months.csv')
# print(dfVisitors.show(100))
# print(dfVisitors.show(100))
print(dfMapMonths.show())
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