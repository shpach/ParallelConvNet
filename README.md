# Parallel Convolutional Neural Network

The goal of this project is to accelerate the forward propagation step of the Convolutional Neural Network (CNN) algorithm using GPUs. The dataset and model are from the [MNIST database](http://yann.lecun.com/exdb/mnist/).

## CNN and MNIST

Provided is a model that has been trained using 60,000 examples (training set images) and the provided test data is 10,000 batched queries (test set images). The expected accuracy of the CNN is `~87%` on the provided test dataset.

The data and model are in [HDF5](https://support.hdfgroup.org/HDF5/) format and we have provided the code to read the input model and the training dataset.


### Running the Serial Code

```{.sh}
./ece408 ../data/test10.hdf5 ../data/model.hdf5 batch_size
```

the `batch_size` must match the size of the dataset. If `batch_size` is unspecified, the default value is dependent on the input (10 for "../data/test10.hdf5", ..., 10000 for "../data/testfull.hdf5"), which is also the size of `data.hdf5`.

## How to Test

Test your implementation with small batch size frist to verify the correctness. You can parse the `data/test100.hdf5` into smaller chunks using your preferred language(e.g. python). 2, 10 and 100 queries are provides in `data/test2.hdf5`, `data/test10.hdf5` and `data/test100.hdf5` in the data folder. Maker sure the data file you feed in has the same batch size as the `batch_size` you specify in the command line.

```{.sh}
./ece408 ../data/test10.hdf5 ../data/model.hdf5 10
```

