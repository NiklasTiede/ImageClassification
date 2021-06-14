
# Image Classification - PyTorch/FashionMNIST

The model predicts categorical data (10 categories of clothing) using the FashionMNIST dataset. Achieved accuracy with the present CNN is 92.7%.

- The `train.py` script trains the weights of the model based on the given hyperparameters
- `tensorboard` is used to compare the yielded accuracies of different hyperparameter optimizations
- the `eval_conf_matrix.py` script creates a confusion matrix to evaluate the trained model in more detail (how well each image category was classified)
