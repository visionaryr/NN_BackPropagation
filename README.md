# Back Propagation Neural Network Trainer

## Overview

This repository presents a complete implementation of a Feed-Forward Neural Network (FNN) trainer written in modern C++.  
The project is structured to offer two distinct implementations for comparison:

1. Single-Threaded Baseline: A robust, sequential implementation of the Backpropagation algorithm.

2. Multi-Threaded Performance Version: A parallelized implementation using a custom Thread Pool.

This structure allows for direct, measurable comparison, demonstrating significant speedup over the single-threaded baseline, particularly for large datasets and batch sizes.

The entire system is designed with a focus on high performance, clean separation of concerns, and robust multi-threading using standard C++11 and later features.

## Default Dataset (MNIST)

This project uses the famous MNIST database of handwritten digits as the default dataset for all training and benchmarking runs. The MNIST dataset is ideal for testing neural network functionality and parallel performance due to its size and simplicity.

Download link: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

## Key Features

- **Dual Implementation for Benchmarking**: Explicit support for both a single-threaded and a multi-threaded training engine.

- **Custom Thread Pool**: A specialized thread pool implementation (ThreadPool class) for managing worker threads and task execution.

- **Parallel Backpropagation**: The training batch is automatically divided into sub-batches, with each sub-batch's forward and backward pass executed concurrently by a worker thread.

- **Efficient Synchronization**: Uses std::mutex and std::lock_guard to protect the global gradient accumulator, ensuring atomic and thread-safe updates with minimal critical sections.

- **Performance Metrics**: Includes accurate timing using std::chrono (Wall Clock Time) to demonstrate the speedup achieved through parallel execution.

- **C++ Matrix Library**: Custom library that utilizes an underlying matrix/vector class for fast linear algebra operations.

- **Open Source Licensed**: Distributed under the permissive MIT License, encouraging reuse and contribution.

## Build and Run

### Prerequisites
- A C++17 compliant compiler (GCC/G++, Clang, or MSVC).

### Branch Structure

| Branch Name | Implementation | Description |
| ----------- | -------------- | ----------- |
|main (or single-thread training) | Single-threaded | The sequential training. |
| MultiThread-Training | Multi-threaded | High-performance version using the Thread Pool.|

## Instructions

### Clone the Repository

```bash
git clone https://github.com/[YourUsername]/NN_BackPropagation.git
cd RepoName
```

### Checkout Desired Branch

```bash
git checkout MultiThread-Training # or main, depending on your setup
```

### Build

```bash
make all
```

### Run

```bash
make run
```

### Build with Debug Mode

```bash
make debug
```

## Configuration and Customization

The project allows users to quickly configure the neural network architecture and the specific subset of the dataset to be trained by modifying two static arrays located in the `main.cpp` file.

### Network Layer Configuration
This array defines the size (number of neurons) of each layer in the Fully Connected Network (FCN). The number of layers is determined by the size of this array.

- The first element must match the input feature size (e.g., 784 for MNIST images).

- The last element must match the number of output categories (e.g., 10 for MNIST digits).

```c
int mNetworkLayout[] = {
  784,  // Input layer
  15,   // Hidden layer
  10    // Output layer
};
```

### Training Category Selection
This array allows you to filter the MNIST dataset to include only specific digits for training and testing. This is useful for binary classification experiments or quick tests on a smaller data subset.

- The values should correspond to the desired target categories (digits 0 through 9).
- The size of this array automatically dictates the size of the output layer. If you set this array to a size $C$, then the Output Layer size in g_LayerConfiguration must be set to $C$.

```c
int mTrainingCategories[] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9
};
```

### Root Path
This project requires a data path to be defined at compile time. The path is where the training and testing dataset is placed. By default, it is configured to use the current working directory where the program is running.

If you need to define an absolute or custom local path for testing or development, you must create a local configuration file `LocalConfig.mk` that Git is instructed to ignore. The sample file is as below,

```
# Define the macro that points to your private header file
LOCAL_PATH_FILE = \"your/data/path/\"
```



## License
This project is licensed under the MIT License.

This is a permissive license that allows anyone to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software. You can find the full terms and conditions in the `LICENSE` file committed to the root of the repository.