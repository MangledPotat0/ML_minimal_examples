# -*- coding: utf-8 -*-

# Built-in module imports
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import typing
from typing import List

# 3rd party module imports
import keras
from keras import Sequential
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, \
        VectorAssembler
from pyspark.pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, StructType, StructField, StringType
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Local module imports

def build_model(input_shape: tuple,
                hidden_sizes: List[int],
                output_shape: int) \
        -> keras.Sequential:
    """
    Builds keras.Sequential model with Input, dense layers, and output layer
    for one-hot encoding (sigmoid+categorical_crossentropy).

    Args:
        input_shape (tuple): Input shape to the sequential model.
        hidden_sizes (List[int]): Size of the hidden dense layers to use.
        output_shape (int): Size of the one-hot encoding vector.

    Returns:
        keras.Sequential: Sequential model.
    """

    model = Sequential()
    model.add(Input(shape=input_shape, name="input"))
    for i, size in enumerate(hidden_sizes):
        model.add(Dense(size, name=f"dense_{i}"))
    model.add(Dense(output_shape, name="output", activation="sigmoid"))
    model.build((None, *input_shape))
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"],
                  optimizer="adam")

    return model

def data_etl(spark) -> DataFrame:
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
    if os.path.isdir("/app/workdir/data/iris.parquet"):
        return spark.read.parquet("/app/workdir/data/iris.parquet")

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
    indexer = StringIndexer(
            inputCols=["class"],
            outputCols=["class_idx"]
    )
    encoder = OneHotEncoder(
            inputCols=["class_idx"],
            outputCols=["class_encoded"],
            dropLast=False
    )
    assembler = VectorAssembler(
            inputCols=["sepal_length",
                       "sepal_width",
                       "petal_length",
                       "petal_width"],
            outputCol="features_prescale"
    )
    scaler = StandardScaler(
            inputCol="features_prescale",
            outputCol="features"
    )

    pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler])

    # Fit data and drop no longer needed columns for memory efficiency
    model = pipeline.fit(data)
    # It's inelegant compared to looping over names but avoids writing data to
    # variable 5 times.
    data = model.transform(data) \
            .drop("sepal_length") \
            .drop("sepal_width") \
            .drop("petal_length") \
            .drop("petal_width") \
            .drop("features_prescale") \
            .drop("class_idx")
    # Load transformed data as parquet so that the same re-processing dosen't
    # have to happen at every run
    data.write.mode("overwrite").parquet("/app/workdir/data/iris.parquet")

    return data

def main(spark: SparkSession):

    # Prepare the data
    train, val, test = data_etl(spark).randomSplit([0.7, 0.05, 0.25],
                                                   seed=42)
    labels = [row.__getattr__("class")
              for row in train.select("class").distinct().collect()]
    train = [np.array(train.select(label).collect()).squeeze()
             for label in ["features", "class_encoded"]]
    val = [np.array(val.select(label).collect()).squeeze()
           for label in ["features", "class_encoded"]]
    test = [np.array(test.select(label).collect()).squeeze()
             for label in ["features", "class_encoded"]]

    # Prepare the model
    input_size = len(train[0][0])
    hidden_sizes = [1]
    output_sizes = len(train[1][0])
    model = build_model((4,), hidden_sizes, output_sizes)
    model.summary()

    # Train the model
    model.fit(
        x=train[0],
        y=train[1],
        validation_data=val,
        batch_size=8,
        epochs=250
    )

    # Test model performance
    test_eval = model.evaluate(
        x=test[0],
        y=test[1]
    )
    predictions = model.predict(test[0])
    cmat = confusion_matrix(np.argmax(test[1],axis=1),
                            np.argmax(predictions, axis=1))
    ax = sns.heatmap(cmat, annot=True, xticklabels=labels, yticklabels=labels)
    plt.ylabel("Ground Truth")
    plt.xlabel("Predictions")
    ax.xaxis.tick_top()
    plt.savefig("confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    spark = SparkSession.builder.appName("torch_petal") \
                                .getOrCreate()
    main(spark)

# EOF
