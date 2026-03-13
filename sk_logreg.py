# -*- coding: utf-8 -*-
"""
sk_logreg.py
Simple script to show the use of sklearn's logistic regression model on the
iris dataset.
"""

# 3rd party module imports
import numpy as np
from pyspark.sql import SparkSession
from sklearn.linear_model import LogisticRegression

# Local module imports
from core.enums import YType
from core.spark_etl import data_etl, numpify
from core.plots import confusion_matrix

def main(spark):
    # Fetch data
    train, test = data_etl(spark, YType.DENSE) \
                            .randomSplit([0.75, 0.25])
    labels = [row.__getattr__("class")
              for row in train.select("class").distinct().collect()]
    train, test = numpify(train, test)

    # Set up and train the regression model
    classifier = LogisticRegression(max_iter=200)
    classifier = classifier.fit(train[0], train[1])

    # Evaluate model
    pred = classifier.predict_proba(test[0])
    confusion_matrix(test[1], np.argmax(pred, axis=1), labels,
                     figname="logreg")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("sklearn").getOrCreate()
    main(spark)

# EOF
