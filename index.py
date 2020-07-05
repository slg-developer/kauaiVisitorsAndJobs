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

# load mapping table to get from FEB to 02, etc...
month_schema = StructType([
    StructField("month_text", StringType())
    , StructField("month_num", StringType())
])
dfMapMonths = spark.read.format("csv")\
    .option("header", True)\
    .schema(month_schema)\
    .load('map_months.csv')
# Load job data
dfJobs = spark.read.format("csv")\
    .option("header", True)\
    .option("infertype", True)\
    .load("kauai_jobs_data.csv")

# Unpivot visitor data so years and moths are in nrows
csv_visitors_pandadf = csv_visitors_pandadf.unstack(level=1).reset_index(name='visitor_count')
# in column level_1 after pivot, 0 = Total Visitor Arrivals, 1 = Domestic
# Visitor Arrivals, 2 = International Visitor Arrivals
# pandas uses NaN for null.  Use .dropna() to drop rows with no visitor count
csv_visitors_clean = csv_visitors_pandadf.loc[ (csv_visitors_pandadf['level_0'] != 'Series') ].dropna()
csv_visitors_clean = csv_visitors_clean.rename(columns={'level_0': 'year_month', 'level_1': 'visitor_type_id'})
# convert pandas dataframe to spark
dfVisitors = spark.createDataFrame(csv_visitors_clean)
# join kauai jobs dataframe with map months so we can join to visitor data on year-months
jobs = dfJobs.alias('jobs')
m = dfMapMonths.alias('m')
dfJ = jobs.join(m, jobs.month == m.month_text, 'inner')
# add column for year as str
dfJ1 = dfJ.withColumn("year_str", col("year").cast(StringType()).substr(1, 4))
#  cast(StringType()).split('.',1)[0])
# add column for year_month
dfJ2 = dfJ1.withColumn("year_month", concat(col("year_str"), lit("-"), col("month_num")))
# make the join to visitor dataframe
vis = dfVisitors.alias('vis')
df_joined = vis.join(dfJ2, vis.year_month == dfJ2.year_month).select(
    "vis.year_month"
    , "visitor_type_id"
    , "visitor_count"
    , "Arts_Entertainment_Recreation"
    , "Accommodation"
    , "FoodServices_DrinkingPlaces"
)