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
| main | Single-threaded | The sequential training. |
| MultiThread-Training | Multi-threaded | High-performance version using the Thread Pool. |

## Instructions

### Clone the Repository

```bash
git clone https://github.com/visionaryr/NN_BackPropagation.git
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

Or run directly with a configuration file(See [Configuration File](#configuration-file) for details):

```bash
./bin/BpProgram -f config.yaml
```

To run with default built-in settings (without a config file):

```bash
./bin/BpProgram
```

### Build with Debug Mode

```bash
make debug
```

## Configuration and Customization

### Configuration File

The project uses a YAML configuration file to specify network architecture, training parameters, and data settings. This allows users to easily experiment with different configurations without recompiling.

To run the program with a configuration file, use the `-f` flag:

```bash
./BpProgram -f config.yaml
```

If no configuration file is specified, the program uses built-in hardcoded defaults.

#### Configuration File Format

The configuration file (`config.yaml`) has three main sections:

**Network Architecture**

Defines the layer sizes for the Fully Connected Network (FCN):

```yaml
Network:
  Layout: [784, 30, 10]  # [input layer, hidden layer(s), output layer]
```

- The first element must match the input feature size (e.g., 784 for MNIST images).
- The last element must match the number of output categories (e.g., 10 for MNIST digits).
- You can have multiple hidden layers by adding more elements.

**Training Parameters**

Configures the backpropagation training algorithm:

```yaml
Training:
  LearningRate: 0.001      # Learning rate (typically 0.001 - 0.1)
  Epochs: 20               # Number of complete passes through the dataset
  TargetLoss: 0.05         # Target loss threshold (training stops early if reached)
  BatchSize: 300           # Number of samples per batch (BATCH_MODE)
  TrainingMode: "BATCH_MODE"  # "BATCH_MODE" or "PATTERN_MODE"
```

**Data Configuration**

Specifies dataset and training categories:

```yaml
Data:
  TrainingCategories: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Digit classes to train on
```

The `TrainingCategories` array determines which digits (0-9) to include in training. The size of this array must match the output layer size in the Network Layout.

### Data Path
This project requires a data path to be defined at compile time. The path is where the training and testing dataset is placed. By default, it is configured to use the current working directory where the program is running.

If you need to define an absolute or custom local path for testing or development, you must create a local configuration file `LocalConfig.mk` that Git is instructed to ignore. The sample file is as below,

```
# Define the macro that points to your private header file
LOCAL_PATH_FILE = \"your/data/path/\"
```

## License
This project is licensed under the MIT License.

This is a permissive license that allows anyone to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software. You can find the full terms and conditions in the `LICENSE` file committed to the root of the repository.