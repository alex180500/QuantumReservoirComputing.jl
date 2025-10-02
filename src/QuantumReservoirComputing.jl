module QuantumReservoirComputing

# Standard library imports
using LinearAlgebra
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
export get_MB, get_mean_last, count_unique_all
export count_unique, unique_indices
export unique_approx, count_unique_approx, unique_indices_approx

# === MATRIX UTILITIES (depends on LinearAlgebra) ===
# Matrix operations and utilities
include("utils/matrices.jl")
export ⊗, kronpow, eigvals_2
export directsum, directsum!, ⊕

# === QUANTUM CONSTRUCTORS (depends on Memoization) ===
# Quantum state vectors and density matrices
include("quantum/constructors.jl")
export ket_0, ket_1, ket_p, ket_m
export rho_0, rho_1, rho_mix
export pauli_x, pauli_y, pauli_z, pauli_p, pauli_m
export H_gate, T_gate, S_gate
export eye2, eye, eye_qubits, max_mixed

# === QUANTUM OPERATOR UTILS (depends on LinearAlgebra) ===
# Utilities for quantum operators
include("quantum/utils_operators.jl")
export get_unitary, isunitary, commute, exp_val

# === QUANTUM STATE UTILS (no external dependencies) ===
# Expectation values, number of qubits and other stuff
include("quantum/utils_states.jl")
export get_nsys, avg_z, avg_x, avg_y, avg_zz
export get_probabilities, get_probabilities!
export get_bloch_vector

# === QUANTUM RANDOM (depends on LinearAlgebra) ===
# Random state generation and Haar measure
include("quantum/random.jl")
export haar_unitary, haar_state, haar_dm
export rand_symmetric_unitary, get_symmetry_blocks

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
include("networks/utils_nets.jl")
export get_link_weight, edges_to_adj, get_order

# === NETWORK METRICS (depends on LinearAlgebra) ===
# network metrics functions
include("networks/metrics.jl")
export degrees, laplacian
export laplacian_spectrum, algebraic_connectivity
export global_clustering

# === QUANTUM NETWORKS (no external dependencies) ===
# Quantum network analysis and correlation networks
include("networks/quantum_nets.jl")
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
export measure, measure!
export measure_probs, measure_probs!
export measure_local, measure_local!
export measure_local_func, measure_local_func!

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
