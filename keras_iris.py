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
from pyspark.pandas import DataFrame
from pyspark.sql import SparkSession
import seaborn as sns

# Local module imports
from core.spark_etl import data_etl, numpify
from core.plots import confusion_matrix

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

def main(spark: SparkSession):

    # Prepare the data
    train, val, test = data_etl(spark, ytype=YType.ONEHOT) \
                            .randomSplit([0.7, 0.05, 0.25], seed=42)
    labels = [row.__getattr__("class")
              for row in train.select("class").distinct().collect()]
    train, val, test = numpify(train, val, test)
    """
    train = [np.array(train.select(label).collect()).squeeze()
             for label in ["features", "class_encoded"]]
    val = [np.array(val.select(label).collect()).squeeze()
           for label in ["features", "class_encoded"]]
    test = [np.array(test.select(label).collect()).squeeze()
             for label in ["features", "class_encoded"]]
    """

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
    confusion_matrix(
            np.argmax(test[1], axis=1),
            np.argmax(predictions, axis=1),
            labels,
            figname="keras_dense"
    )

if __name__ == "__main__":
    spark = SparkSession.builder.appName("torch_petal") \
                                .getOrCreate()
    main(spark)

# EOF
