# QuantumReservoirComputing.jl

<div align="center">
    <img src="assets/mbl.png" width="25%" alt="many body quantum localization">
    &nbsp;&nbsp;&nbsp;
    <img src="assets/mnist.png" width="40%" alt="mnist images pca">
    &nbsp;&nbsp;&nbsp;
    <img src="assets/phases.png" width="22%" alt="qrc dynamical phases">
</div>

Quantum Reservoir Computing and Quantum Extreme Learning Machine package written in Julia. This package provides tools for quantum reservoir computing (QRC) and quantum extreme learning machines (QELM), including:

- Quantum states, measurements, correlations
- Very fast partial trace interface for qubits and general multipartite systems
- Various quantum utils functions, quantum encodings and quantum hamiltonians
- QRC and QELM algorithms
- Simple neural networks via [Flux.jl](https://fluxml.ai/Flux.jl/stable/)
- Some simple complex network and graph theory tools

> [!IMPORTANT]
> This package is under active and early development. The API may change frequently, some feature are not yet implemented, and the documentation is still a work in progress. If you have any questions feel free to [open an issue](https://github.com/alex180500/QuantumReservoirComputing.jl/issues/new/choose).

## Example Usage

Here's a complete example demonstrating how to use the library for quantum extreme learning with Haar random unitaries on the MNIST dataset:

```julia
using QuantumReservoirComputing
using MLDatasets: MNIST

# Step 1: Load MNIST data
train_x, train_y = MNIST(:train)[:]
test_x, test_y = MNIST(:test)[:]

# Reshape and normalize data
train_data = reshape(train_x, 784, :) ./ 255.0
test_data = reshape(test_x, 784, :) ./ 255.0

# Step 2: Apply PCA and create quantum states
N = 8  # number of qubits (2^8 = 256 dimensions)
n_components = 2^N

# Apply PCA to reduce dimensionality
train_pca, test_pca = pca_analysis(train_data, test_data, n_components)

# Convert to quantum states using amplitude encoding
function amplitude_encode(data::AbstractMatrix{T}) where {T<:Real}
    quantum_states = similar(data, ComplexF64)
    @inbounds for col in axes(data, 2)
        # Normalize each column to create a valid quantum state
        norm_factor = sqrt(sum(abs2, view(data, :, col)))
        if norm_factor > 0
            quantum_states[:, col] .= view(data, :, col) ./ norm_factor
        else
            quantum_states[:, col] .= 0
        end
    end
    return quantum_states
end

quantum_train = amplitude_encode(train_pca)
quantum_test = amplitude_encode(test_pca)
quantum_states = hcat(quantum_train, quantum_test)

# Step 3: Create Haar random unitary as quantum reservoir
base_U = haar_unitary(2^N)

# Step 4: Compute reservoir outputs using quantum evolution
reservoir_outputs = qelm_compute(base_U, quantum_states)

# Step 5: Train neural network classifier
train_labels = train_y .+ 1  # Convert to 1-based indexing
test_labels = test_y .+ 1

results = nn_layer(
    Float32.(reservoir_outputs),
    train_labels,
    test_labels,
    rate=0.1,
    epochs=100
)

println("Final training accuracy: $(results.train_accuracies[end])")
println("Final test accuracy: $(results.test_accuracies[end])")
```

## License

This package is distributed under [Apache-2.0 License](LICENSE). If you use my code extensively I would greatly appreciate if you could credit me, [**Alessandro Romancino**](https://orcid.org/0009-0004-2812-6251), or my GitHub profile [`https://github.com/alex180500`](https://github.com/alex180500). _Thanks!_
