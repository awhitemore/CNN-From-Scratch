# CNN from Scratch in C++

A complete implementation of a Convolutional Neural Network built entirely from scratch in C++ for Fashion-MNIST classification, demonstrating deep understanding of neural network fundamentals without relying on high-level frameworks.

## 🎯 Project Motivation

While modern deep learning is dominated by high-level frameworks like TensorFlow and PyTorch, I built this CNN from the ground up to:

- **Understand the Mathematics**: Implement backpropagation, convolution operations, and gradient calculations by hand
- **Master Low-Level Programming**: Work directly with memory management, tensor operations, and numerical computing in C++
- **Bridge Theory and Practice**: Transform mathematical concepts from research papers & class into working code
- **Demonstrate Systems Thinking**: Build a complete machine learning pipeline from data loading to model evaluation

This project represents a commitment to understanding not just *how* to use neural networks, but *why* they work.

## 🏗️ Architecture

The CNN implements a classic architecture optimized for Fashion-MNIST classification:

```
Input (28×28 grayscale images)
    ↓
Convolutional Layer (8 filters, 3×3 kernels, ReLU activation)
    ↓
Relu
    ↓
Max Pooling (2×2)
    ↓
Flatten (13×13×8 = 1,352 features)
    ↓
Dense Layer (1,352 → 10 neurons)
    ↓
Softmax Activation
    ↓
Output (10 fashion classes)
```

## 📊 Performance Results

Training on Fashion-MNIST (60,000 training images, 10,000 test images):

| Metric | Result |
|--------|--------|
| **Final Test Accuracy** | **85.77%** |
| **Training Accuracy** | 86.37% |
| **Convergence** | 3 epochs |
| **Generalization Gap** | 0.6% (minimal overfitting) |

**Learning Progression:**
- Epoch 1: 83.07% → Epoch 3: 86.37%
- Loss reduction: 4.24 → 0.0026 (1000x improvement)
- Consistent improvement with stable convergence

## 🔧 Technical Implementation

### Core Components Built from Scratch:

- **Custom Tensor Class**: 3D tensor implementation 
- **Convolution Layer**: Forward and backward pass with learnable 3×3 filters
- **Max Pooling**: 2×2 downsampling with gradient tracking for backpropagation  
- **Dense Layer**: Fully connected layer with weight updates via gradient descent
- **Softmax Activation**: Stable probability distribution for multiclass classification
- **Complete Backpropagation**: End-to-end gradient computation and weight updates

### Key Engineering Features:

- **Multi-epoch Training**: Configurable training cycles with detailed progress tracking
- **Cross-entropy Loss**: Proper loss function with numerical stability (log clamping)
- **Binary Data Loading**: Direct parsing of MNIST IDX file format with endianness handling
- **Memory Efficient**: Careful resource management for processing 60K+ images
- **Reproducible Results**: Fixed random seed ensures consistent 85.77% accuracy

## 💼 Professional Skills Demonstrated

### **Systems Programming & C++**
- Manual memory management and efficient data structures
- Binary file I/O and bit manipulation for MNIST format parsing
- Performance optimization with compiler flags and algorithmic efficiency
- Clean object-oriented design with proper encapsulation

### **Mathematical Implementation**
- Matrix operations and multidimensional tensor mathematics
- Gradient computation using chain rule across complex network architectures
- Numerical stability techniques (overflow prevention, log clamping)
- Understanding of convolution mathematics and backpropagation calculus

### **Machine Learning Engineering**
- Complete ML pipeline from raw data to trained model evaluation
- Hyperparameter management and systematic training procedures
- Model performance analysis and convergence monitoring
- Cross-validation methodology and overfitting detection

### **Software Architecture & DevOps**
- Modular design enabling easy extension and maintenance
- Professional build system with Makefile and dependency management  
- Version control best practices and documentation
- Reproducible development environment setup

## 📦 Setup and Usage

### Prerequisites
- C++ compiler with C++17 support (GCC 7+ or Clang 5+)
- Make utility
- curl (for downloading data)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/cnn-from-scratch.git
   cd cnn-from-scratch
   ```

2. **Download Fashion-MNIST data:**
   ```bash
   make data
   ```

3. **Compile and run:**
   ```bash
   make
   ./cnn
   ```

### Expected Output
```
=== Epoch 1/3 ===
  Sample 0, Loss: 4.2413
  Predicted: Trouser, Actual: Ankle boot
  ...
Epoch 1 - Train Accuracy: 0.830733, Average Loss: 0.485023

=== Epoch 2/3 ===
  ...
Final Test Accuracy: 0.8577
```

## 📁 Project Structure

```
cnn-from-scratch/
├── README.md              # This documentation
├── Makefile              # Build configuration with optimization
├── main.cpp              # Training loop and MNIST data loading
├── tensor.h/cpp          # Custom 3D tensor implementation
├── conv_layer.h/cpp      # Convolutional and max pooling layers
├── dense.h/cpp           # Fully connected layer with backpropagation
├── softmax.h             # Softmax activation function
└── .gitignore           # Excludes data files and binaries
```

## 🔄 Architecture Decisions

### **Why C++ Over Python?**
- **Understanding**: Creating this CNN without the aid of standard ML libraies allowed me to learn the underworkings of this project

### **Why This Architecture?**
- **Simplicity**: Focuses on core CNN concepts without unnecessary complexity
- **Educational**: Each component is implementable and understandable from first principles
- **Effective**: Achieves strong performance (85%+) on a standard benchmark
- **Extensible**: Clean design allows easy addition of new layer types

## 🎓 Key Technical Learnings

Building this CNN reinforced that modern frameworks abstract away crucial implementation details:

- **Gradient Flow**: How gradients propagate through pooling layers and tensor reshaping
- **Memory Layout**: How tensor storage affects computational efficiency
- **Numerical Stability**: Why techniques like log clamping prevent training failures  
- **Optimization Dynamics**: How learning rates and weight initialization affect convergence

This foundation enables more effective debugging, optimization, and architecture design when using any ML framework.

## 🔍 Code Quality Features

- **Reproducible Results**: Fixed random seed (42) ensures consistent output
- **Error Handling**: Comprehensive file I/O and format validation
- **Professional Build**: Optimized compilation with proper dependency management
- **Clean Interfaces**: Well-defined class boundaries and minimal coupling
- **Documentation**: Inline comments explaining mathematical operations

## 🌟 Future Enhancements

While the current implementation achieves its educational and performance goals, potential extensions include:

- **Additional Layers**: Batch normalization, dropout, additional conv layers
- **Advanced Optimizers**: Adam, RMSprop with momentum
- **GPU Acceleration**: CUDA implementation for large-scale training
- **Model Persistence**: Save/load trained models for deployment
- **Data Augmentation**: Rotation, scaling, noise injection for improved generalization

---

*This project represents 40+ hours of implementation, debugging, and optimization work, demonstrating commitment to understanding machine learning at a fundamental level. The resulting expertise enables effective work with any ML framework or custom implementation requirements.*
