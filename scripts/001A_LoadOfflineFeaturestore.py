## This file loads the featurestore with test and training data. It is assumed that the feast feature specifications are already created
## in featurespec.py file in homecredit_feature_repo
##PLEASE note that if the features change then the feature store load will not work. use the homecredit_build_featurestore.ipynb file to 
## rebuild the features of the featurestore and update the feature_spec.py file in homecredit_feature_repo

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.sql.functions import * 
import unicodedata
import re
import os
from pyspark.sql.window import *


## SET UP CODE

#db_path = "s3a://tsel-bucket/data/warehouse/tablespace/managed/hive/"
storage_path = os.environ["STORAGE_PATH"]
db_path=storage_path  + "data/warehouse/tablespace/managed/hive/"
print(db_path)

#Load Training and Test Data Path Location, required for feature store data load separation between training and test data
datasource_dict = { "train" : {}, 
                    "test": {}
                  }
datasource_dict["train"]["path"]  = os.environ["DATASOURCE_PATH_TRAIN"]
datasource_dict["test"]["path"] = os.environ["DATASOURCE_PATH_TEST"]
datasource_dict["train"]["sourcedatatype"] = "train"
datasource_dict["test"]["sourcedatatype"] = "test"



def normalize(column: str) -> str:
    """
    Normalize column name by replacing invalid characters with underscore
    strips accents and make lowercase
    :param column: column name
    :return: normalized column name
    """
    n = re.sub(r"[ ,;:{}()/\n\t=\-\+]+", '_', column.lower())
    return unicodedata.normalize('NFKD', n).encode('ASCII', 'ignore').decode()

def build_feature_offlinestore(source, sourcedatatype ):
    """
    Build the offline featurestore for Training and Test Data. Feast uses point in time capture to serve data for training and inference
    Notes:
    1. Since the data has not timestamp per record, one is artificially generated for the featurestore
    2. The Training data is stored with current datetimestamp - 2 months and each record is then saved with a decreased interval of 
       10 seconds. The test data is saved 2 years in the future with an incremental interval of time records. 
    3. This logic is then used in fetching the features for training and testing.
    """

    homecredit_raw_df=spark.read.option("header", True) \
                            .option("inferSchema",True) \
                            .csv(source)
    # using the function
    homecredit_df = homecredit_raw_df.toDF(*map(normalize, homecredit_raw_df.columns))
    homecredit_df=homecredit_df.drop(col("_c0"))
    homecredit_df = homecredit_df.drop(col("index"))
    homecredit_df = homecredit_df.withColumn("monotonically_increasing_id", monotonically_increasing_id())
    window = Window.orderBy(col('monotonically_increasing_id'))
    homecredit_df = homecredit_df.withColumn('increasing_id', row_number().over(window))

    ## cast to float
    cast_cols =["new_phone_to_birth_ratio_employer", "prev_app_credit_perc_max", "refused_app_credit_perc_max", "instal_payment_perc_max", "instal_payment_perc_min" ]
    for col_name in cast_cols:
        homecredit_df = homecredit_df.withColumn(col_name, col(col_name).cast('double'))

    ## remove transformation columns
    homecredit_df = homecredit_df.withColumn("created", current_timestamp())
    if sourcedatatype == "train":
        homecredit_df = homecredit_df.withColumn("event_timestamp", current_timestamp() - expr("INTERVAL 2 Month")  - expr("INTERVAL 10 seconds") * col("increasing_id"))  
    elif sourcedatatype == "test":
        homecredit_df = homecredit_df.withColumn("event_timestamp", current_timestamp() + expr("INTERVAL 2 Years")  + expr("INTERVAL 10 seconds") * col("increasing_id")) 
    else:
        pass
    
    homecredit_df =homecredit_df.drop(col("monotonically_increasing_id"))
    homecredit_df = homecredit_df.drop(col("increasing_id")) 

    
    return homecredit_df
    

# Finally Spark work
spark = (SparkSession
    .builder
    .appName("homecredit-spark")
    .config("spark.sql.warehouse.dir", db_path)
    .config("spark.hadoop.fs.s2a.s3guard.ddb.region", "us-east-1")
    .config("spark.kerberos.access.hadoopFileSystem",storage_path)
    .config('spark.hadoop.fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider')
    .master("local[5]") # should be possible to change this to SPARK on Yarn or SPARK on Kubernetes
    .getOrCreate())

homecredit_train_df = build_feature_offlinestore(datasource_dict["train"]["path"], datasource_dict["train"]["sourcedatatype"])
homecredit_test_df =  build_feature_offlinestore(datasource_dict["test"]["path"], datasource_dict["test"]["sourcedatatype"])

# join the two datasets to build the complete dataframe
homecredit_df = homecredit_train_df.union(homecredit_test_df)

# write to hive table but this can be any datasink for SPARK                                       
homecredit_df.write \
    .mode("overwrite") \
    .format("parquet") \
    .saveAsTable("homecredit_processed_data")
    
print((homecredit_train_df.count(), len(homecredit_train_df.columns)))
print((homecredit_test_df.count(), len(homecredit_test_df.columns)))
print((homecredit_df.count(), len(homecredit_df.columns)))


##PLEASE note that if the features change then the feature store load will not work. use the homecredit_build_featurestore.ipynb file to 
## rebuild the features of the featurestore and update the feature_spec.py file in homecredit_feature_repo