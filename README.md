# Simplest code for deploying keras model

## How to try the code out
Build the Docker container using `setup.sh`. When the container is built, run `launch.sh` to enter the work environment. Then, run `python keras_iris.py`.

## Some discussions
This model can have arbitrary number of hidden layers (keras `Dense` layers) of arbitrary sizes, but the default implementation in this codebase has just one hidden layer with just one neuron (this is enough to get pretty good accuracy).

Here's the result when the code is run as-is without modification:
![confusion matrix](https://raw.githubusercontent.com/MangledPotat0/ML_minimal_examples/refs/heads/main/confusion_matrix.png)
