# -*- coding: utf-8 -*-
"""
core/spark_etl.py
ETL workflow written in pyspark
"""

# Built-in module imports
import os

# 3rd party module imports
import numpy as np
from numpy import ndarray
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, \
        VectorAssembler
from pyspark.pandas import DataFrame
from pyspark.sql.types import FloatType, StringType, StructField, StructType
from pyspark.sql import SparkSession

# Local module imports
from core.enums import YType

def data_etl(spark: SparkSession, ytype: YType) -> DataFrame:
    """
    Performs ETL on the iris dataset from archive.ics.uci.edu/dataset/53/iris.
    Since iris.data file does not contain header row, a custom defined schema
    is used. After extraction, data transformation is performed where the four
    features are turned into a feature vector and scaled, and the class labels
    are one-hot encoded. Then, the output of the transformation is loaded to
    iris.parquet.

    If iris.parquet already exists, all steps are skipped and iris.parquet is
    read directly.

    Args:
        spark (SparkSession): Spark session handle for the process.

    Returns:
        spark.pandas.DataFrame: A DataFrame object containing transformed data.
    """

    # Bypass the ETL stages and read parquet directly if exists
    if os.path.isdir(f"/app/workdir/data/iris_{ytype.name}.parquet"):
        return spark.read.parquet(
                f"/app/workdir/data/iris_{ytype.name}.parquet"
        )

    # Make custom schema because the data file does not have a header
    schema = StructType([
        StructField("sepal_length", FloatType(), True),
        StructField("sepal_width", FloatType(), True),
        StructField("petal_length", FloatType(), True),
        StructField("petal_width", FloatType(), True),
        StructField("class", StringType(), True)
    ])

    # Load file
    data = spark.read.format("csv").schema(schema).option("header", False) \
            .load("/app/workdir/data/iris.data")

    # Prepare data transformation stages
    stages = []
    match ytype:
        case YType.ONEHOT:
            print("OOOO")
            #Indexer
            stages.append(StringIndexer(
                    inputCols=["class"],
                    outputCols=["class_idx"]
            ))
            #Encoder
            stages.append(OneHotEncoder(
                    inputCols=["class_idx"],
                    outputCols=["class_encoded"],
                    dropLast=False
            ))
        case YType.DENSE:
            #Indexer
            stages.append(StringIndexer(
                    inputCols=["class"],
                    outputCols=["class_encoded"]
            ))
    
    # VectorAssembler
    stages.append(VectorAssembler(
            inputCols=["sepal_length",
                       "sepal_width",
                       "petal_length",
                       "petal_width"],
            outputCol="features_prescale"
    ))
    #Scaler
    stages.append(StandardScaler(
            inputCol="features_prescale",
            outputCol="features"
    ))

    pipeline = Pipeline(stages=stages)

    # Fit data and drop no longer needed columns for memory efficiency
    model = pipeline.fit(data)
    # It's inelegant compared to looping over names but avoids writing data to
    # variable 5 times.
    try:
        data = model.transform(data) \
                .drop("sepal_length") \
                .drop("sepal_width") \
                .drop("petal_length") \
                .drop("petal_width") \
                .drop("features_prescale") \
                .drop("class_idx")
    except ValueError:
        data = model.transform(data) \
                .drop("sepal_length") \
                .drop("sepal_width") \
                .drop("petal_length") \
                .drop("petal_width") \
                .drop("features_prescale") \
    # Load transformed data as parquet so that the same re-processing dosen't
    # have to happen at every run
    data.write.mode("overwrite").parquet("/app/workdir/data/iris.parquet")

    return data

def numpify(*args):
    """
    Convert the dataframe columns into list of arrays for compatibility
    """
    return [
        [
            np.array(arg.select(label).collect()).squeeze()
            for label in ["features", "class_encoded"]
        ]
        for arg in args
    ]


# EOF
