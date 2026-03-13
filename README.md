# Simplest code for deploying keras model

## How to try the code out
Build the Docker container using `setup.sh`. When the container is built, run `launch.sh` to enter the work environment. Then, choose one of the following:

1. To run the sequential mdoel with dense hidden layers, use `python keras_iris.py`.
2. To run the sklearn LogisticRegression model, use `python sk_logreg.py`.

The point of this codebase is not to do cutting edge project, but to show my coding style. What is shown in this simple script is my current style of writing code WITHOUT any AI tool, not even autocompletion. I wrote the code using vim inside the Docker container included in this repo and you will find no additional help in it other than the built-in default syntax highlighting.

## Some discussions
This model can have arbitrary number of hidden layers (keras `Dense` layers) of arbitrary sizes, but the default implementation in this codebase has just one hidden layer with just one neuron (this is enough to get pretty good accuracy).

Here's the result when the code is run as-is without modification.

Using keras sequential model:  
![sequential model confusion matrix](https://raw.githubusercontent.com/MangledPotat0/ML_minimal_examples/refs/heads/main/keras_dense_confusion_matrix.png)

Using sklearn LogisticRegression model:  
![sklearn logreg confusion matrix](https://raw.githubusercontent.com/MangledPotat0/ML_minimal_examples/refs/heads/main/logreg_confusion_matrix.png)
