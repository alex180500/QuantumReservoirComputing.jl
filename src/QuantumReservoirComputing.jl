module QuantumReservoirComputing

# Standard library imports
using LinearAlgebra: Diagonal, Hermitian, I, copyto!, diag, eigen, eigvals, kron, kron!, qr!
using Random: rand!
using Base.Threads: @threads

# External packages
using Distributions: Categorical, Multinomial, Uniform
using Flux: Flux, Adam, Chain, Dense
using Memoization: @memoize
using MultivariateStats: PCA, fit, transform
using ProgressMeter: Progress, next!
using StatsBase: countmap, mean, std
using TensorOperations: tensortrace

# === BIT UTILITIES (no external dependencies) ===
# Bit manipulation utilities
include("utils/bits.jl")
export get_bit, get_bit_table

# === GENERIC UTILITIES (depends on StatsBase) ===
# Generic utility functions, faster than StatsBase
include("utils/generic.jl")
export get_mb, get_mean_last, get_average_data
export count_unique, unique_indices
export unique_approx, count_unique_approx, unique_indices_approx

# === MATRIX UTILITIES (depends on LinearAlgebra) ===
# Matrix operations and utilities
include("utils/matrices.jl")
export direct_sum, ⊕, kron_pow, ⊗, eigvals_2

# === QUANTUM CONSTANTS (no external dependencies) ===
# Quantum state vectors and density matrices
include("quantum/constants.jl")
export ket_0, ket_1, ket_p, ket_m
export rho_0, rho_1, rho_mix, rho_p, rho_m
export sig_x, sig_y, sig_z, sig_p, sig_m, eye2

# === QUANTUM UTILITIES (depends on Memoization) ===
# System utilities and random state generation
include("quantum/q_utils.jl")
export get_nsys, max_mixed, eye, eye_qubits, commute

# === QUANTUM RANDOM (depends on LinearAlgebra) ===
# Random state generation and Haar measure
include("quantum/random.jl")
export haar_unitary, haar_state, haar_dm
export rand_symmetric_unitary, rand_symmetric_unitary!, get_symmetry_blocks

# === QUANTUM OPERATORS (no external dependencies) ===
# Measurement operators and expectation values
include("quantum/operators.jl")
export avg_z, avg_x, avg_y, avg_zz
export get_unitary, get_probabilities, get_probabilities!
export get_bloch_vector

# === PARTIAL OPERATIONS (depends on Base.Cartesian, TensorOperations) ===
# Partial trace operations
include("quantum/partial.jl")
export ptrace, ptrace_qubits, ptrace_2qubits
export ptrace_d, ptrace_qubits_d

# === QUANTUM CORRELATIONS (depends on LinearAlgebra) ===
# Quantum correlations and entanglement
include("quantum/correlations.jl")
export concurrence, vn_entropy, mutual_info

# === LOCAL OPERATORS (depends on LinearAlgebra) ===
# Local operators used for hamiltonians
include("quantum/local.jl")
export LocalOperators

# === NETWORK THEORY (depends on LinearAlgebra) ===
# Basic network operations and graph theory
include("networks/networks.jl")
export get_link_weight, edges_to_adj, get_order
export degrees, laplacian

# === NETWORK METRICS (depends on LinearAlgebra) ===
# network metrics functions
include("networks/metrics.jl")
export laplacian_spectrum, algebraic_connectivity

# === QUANTUM NETWORKS (no external dependencies) ===
# Quantum network analysis and correlation networks
include("networks/q_networks.jl")
export node_entropies!, correlation_network
export correlation_edgelist, correlation_edgelist!

# === QRC ENCODINGS (depends on LinearAlgebra) ===
# encodings functions for quantum reservoir computing
include("qrc/encodings.jl")
export encode_qubit, dense_angle_encoding

# === QRC HAMILTONIANS (depends on Distributions) ===
# Hamiltonian constructors for quantum reservoir computing
include("qrc/hamiltonians.jl")
export h_monroe, xx_monroe_obc, xx_monroe_pbc, z_noisy

# === QRC MEASUREMENTS (depends on Distributions) ===
# Quantum measurement protocols and statistics
include("qrc/measurements.jl")
export local_measure, local_measure!
export montecarlo_measure, montecarlo_measure!
export simulated_measure, simulated_measure!
export get_binary_outcomes!

# === QRC TOOLS (depends on Base.Threads) ===
# Simulations function for QELM for pure states
include("qrc/qrc_tools.jl")
export qelm_compute, qelm_compute_networks

# === MACHINE LEARNING FUNCTIONS (depends on Flux, MultivariateStats, StatsBase) ===
# Data analysis and preprocessing
include("ml/functions.jl")
export pca_analysis, rescale_data, accuracy

# === MACHINE LEARNING LINEAR (depends on Flux, ProgressMeter) ===
# Neural network training utilities
include("ml/linear.jl")
export train_epoch!, nn_layer

# === MACHINE LEARNING PROGRESS (depends on ProgressMeter) ===
# Training progress and monitoring
include("ml/progress.jl")
export ml_showvalues, @trainprogress

end # module QuantumReservoirComputing
