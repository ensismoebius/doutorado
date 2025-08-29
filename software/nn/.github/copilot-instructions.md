# Neural Network Framework Guidelines

This is a C++ neural network framework focused on spiking neural networks and autoencoder implementations. Below are the key patterns and conventions to follow when working with this codebase.

## Project Structure

```
nn/
├── src/
│   ├── initializers/     # Weight initialization strategies
│   ├── layers/           # Neural network layers
│   ├── optimizers/       # Optimization algorithms
│   ├── tensor/          # Tensor operations and data structures
│   └── util/            # Utility functions and helpers
├── lib/                 # External libraries
└── build/              # Build artifacts
```

## Key Components

### 1. Tensor System

- Base data structure is `Tensor` in `tensor/Tensor.hpp`
- Uses Eigen for matrix operations
- Each tensor contains both `data` and `grad` matrices

### 2. Layer Architecture

- All layers inherit from `Module` base class
- Key methods to implement:
  - `forward(const Tensor &input) -> Tensor`
  - `backward(const Tensor &grad_output) -> Tensor`
- Example layers: `Linear`, `Leaky`, `ReLU`, `LeakyReLU`

### 3. Model Construction

- Use `Sequential` container for model composition
- PyTorch-like syntax: `Sequential({layer1, layer2, ...})`
- Supports both training and inference

### 4. Initialization

- Use appropriate initializers from `initializers/`
  - `kaimingSNNInitializer` for spiking networks
  - `xavierInitializer` for standard networks

### 5. Optimization

- Optimizers inherit from `Optimizer` base class
- Available optimizers: `Adam`, `SGD`, `SGDMinimal`
- Use `optimizer.attach(params)` to connect parameters

## Development Workflows

### Building

```bash
mkdir build && cd build
cmake ..
make
```

### Testing

- Unit tests use Google Test framework
- Tests organized by component:
  - `layers_gtest.cpp`
  - `optimizers_gtest.cpp`
  - `initializers_gtest.cpp`

### OpenMP Support

- Parallelization enabled for batch processing
- Use `#pragma omp parallel for` for parallel operations

## Conventions

1. **Code Style**

   - Use `auto` for return types with trailing return syntax
   - Member variables use snake_case
   - Functions/methods use camelCase

2. **Memory Management**

   - Use `std::shared_ptr` for layer ownership
   - Tensor operations are copy-based for safety

3. **Error Handling**

   - Use assertions for internal checks
   - Return values should be validated explicitly

4. **Performance**
   - Optimize matrix operations with Eigen
   - Use OpenMP for batch parallelization
   - Avoid unnecessary tensor copies

## Integration Points

1. **Data Loading**

   - Use `synthetic_spike_data.hpp` for spike train generation
   - Support for batch processing via `batching.hpp`

2. **Model Saving/Loading**
   - Use `NnSaver.hpp` for model persistence
   - Supports saving weights and architecture

## Common Tasks

1. **Creating a New Layer**

   ```cpp
   struct NewLayer : public Module {
     auto forward(const Tensor &input) -> Tensor override;
     auto backward(const Tensor &grad_output) -> Tensor override;
   };
   ```

2. **Training Loop Structure**
   ```cpp
   optimizer.zero_grad(params);
   Tensor output = model.forward(input);
   Tensor loss = loss_function(output, target);
   Tensor grad = loss_function.backward(output);
   model.backward(grad);
   optimizer.step(params);
   ```
