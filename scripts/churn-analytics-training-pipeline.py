from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import os
import shutil
import csv

import boto3

from awsglue.utils import getResolvedOptions

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, FloatType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql.functions import col
from mleap.pyspark.spark_support import SimpleSparkSerializer

def csv_line(data):
    s = str(data[0])
    a = data[1].toArray() 
    for i in range(len(a)) :
        s += ','+str(a[i])
    return s

def main():
    spark = SparkSession.builder.appName("churn-analytics").getOrCreate()
    
    args = getResolvedOptions(sys.argv, ['S3_INPUT_BUCKET',
                                         'S3_INPUT_KEY_PREFIX',
                                         'S3_OUTPUT_BUCKET',
                                         'S3_OUTPUT_KEY_PREFIX', 
                                         'S3_MODEL_BUCKET',
                                         'S3_MODEL_KEY_PREFIX'])
    
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    spark.sparkContext._jsc.hadoopConfiguration().set("mapred.output.committer.class",
                                                      "org.apache.hadoop.mapred.FileOutputCommitter")

    # Defining the schema corresponding to the input data. The input data does not contain the headers
    callstats_schema = StructType([StructField('State', StringType(), True),
                                   StructField('AccountLength', IntegerType(), True),
                                   StructField('AreaCode', IntegerType(), True),
                                   StructField('Phone', StringType(), True),
                                   StructField('IntlPlan', StringType(), True),
                                   StructField('VMailPlan', StringType(), True),
                                   StructField('VMailMessage', IntegerType(), True),
                                   StructField('DayMins', FloatType(), True),
                                   StructField('DayCalls', IntegerType(), True),
                                   StructField('DayCharge', FloatType(), True),
                                   StructField('EveMins', FloatType(), True),
                                   StructField('EveCalls', IntegerType(), True),
                                   StructField('EveCharge', FloatType(), True),           
                                   StructField('NightMins', FloatType(), True),
                                   StructField('NightCalls', IntegerType(), True),
                                   StructField('NightCharge', FloatType(), True),           
                                   StructField('IntlMins', FloatType(), True), 
                                   StructField('IntlCalls', IntegerType(), True),  
                                   StructField('IntlCharge', FloatType(), True),  
                                   StructField('CustServCalls', IntegerType(), True),
                                   StructField('Churn?', StringType(), True)])

    # Downloading the data from S3 into a Dataframe
    raw_df = spark.read.csv(('s3://' + os.path.join(args['S3_INPUT_BUCKET'], args['S3_INPUT_KEY_PREFIX'],
                                   'churn.csv')), header=False, schema=callstats_schema)
    
    categoricalColumns = ["State", "AreaCode", "IntlPlan", "VMailPlan"]
    stages = [] # stages in our Pipeline

    for categoricalCol in categoricalColumns :
        idxName = categoricalCol+"Idx"
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=idxName)
        catVec = categoricalCol+"Vec"
        encoder = OneHotEncoder(inputCol=idxName, outputCol=catVec, dropLast=False)
        stages += [stringIndexer, encoder]
        
    numericCols = ["AccountLength","VMailMessage","DayMins","DayCalls","EveMins","EveCalls","NightMins",
                   "NightCalls","IntlMins","IntlCalls", "CustServCalls"]

    assemblerInputs = numericCols+[c + "Vec" for c in categoricalColumns]
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]
    
    mlPipeline = Pipeline().setStages(stages)
    pipelineModel = mlPipeline.fit(raw_df)
    dataset = pipelineModel.transform(raw_df).select('*',col('Churn?').contains('True').cast('integer').alias('labels'))
    
    # Split the overall dataset into 80-20 training and validation
    (train_df, test_df) = dataset.randomSplit([0.8, 0.2])

    # Convert the train dataframe to RDD to save in CSV format and upload to S3
    train_rdd = train_df.rdd.map(lambda r: (r.labels, r.features))
    train_lines = train_rdd.map(csv_line)
    train_lines.saveAsTextFile('s3://' + os.path.join(args['S3_OUTPUT_BUCKET'], args['S3_OUTPUT_KEY_PREFIX'], 'train'))
    
    # Convert the validation dataframe to RDD to save in CSV format and upload to S3
    test_rdd = test_df.rdd.map(lambda r: (r.labels, r.features))
    test_lines = test_rdd.map(csv_line)
    test_lines.saveAsTextFile('s3://' + os.path.join(args['S3_OUTPUT_BUCKET'], args['S3_OUTPUT_KEY_PREFIX'], 'test'))
    
    # Serialize and store the model via MLeap  
    SimpleSparkSerializer().serializeToBundle(pipelineModel, "jar:file:/tmp/model.zip", test_df)
    
    # Unzip the model as SageMaker expects a .tar.gz file but MLeap produces a .zip file
    import zipfile
    with zipfile.ZipFile("/tmp/model.zip") as zf:
        zf.extractall("/tmp/model")

     # Writw back the content as a .tar.gz file
    import tarfile
    with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
        tar.add("/tmp/model/bundle.json", arcname='bundle.json')
        tar.add("/tmp/model/root", arcname='root')
    
    # Upload the model in tar.gz format to S3 so that it can be used with SageMaker for inference later
    s3 = boto3.resource('s3') 
    file_name = os.path.join(args['S3_MODEL_KEY_PREFIX'], 'model.tar.gz')
    s3.Bucket(args['S3_MODEL_BUCKET']).upload_file('/tmp/model.tar.gz', file_name)

if __name__ == "__main__":
    main()
