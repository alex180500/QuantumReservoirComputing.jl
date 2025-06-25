module QuantumReservoirComputing

# Standard library imports
using LinearAlgebra:
    qr!, diag, Diagonal, eigvals, Hermitian, I, eigen, kron!, kron
using Random: AbstractRNG, default_rng, rand!
using Statistics: mean, std
using Base.Threads: @threads

# External packages
using Distributions: Categorical, Uniform, Multinomial
using Flux
using Memoization: @memoize
using MultivariateStats: PCA, fit, transform
using ProgressMeter: Progress, next!
using TensorOperations: tensortrace

# === UTILITIES (foundation Layer - no dependencies) ===
# utils functions for bit operations
include("utils/bits.jl")
export get_bit, get_bit_table

# some generic utils functions
include("utils/generic.jl")
export get_mb, count_unique, count_unique_dict
export get_mean_last, get_average_data

# some matrix utils function
include("utils/matrices.jl")
export ⊗, ⊕, direct_sum, eigvals_2

# === QUANTUM CONSTANTS (foundation layer) ===
# constants useful for qubits, common operators
include("quantum/constants.jl")
export ket_0, ket_1, ket_p, ket_m
export rho_0, rho_1, rho_mix, rho_p, rho_m
export sig_x, sig_y, sig_z, sig_p, sig_m, eye2

# === QUANTUM UTILITIES (depends on constants) ===
# quantum utils
include("quantum/q_utils.jl")
export get_nsys, max_mixed, eye, eye_qubits
export haar_unitary, haar_state, haar_dm

# === QUANTUM OPERATORS (depends on constants and utils) ===
# quantum operators functions
include("quantum/operators.jl")
export avg_z, avg_x, avg_y, avg_zz
export get_unitary, get_probabilities, get_probabilities!
export get_bloch_vector

# === PARTIAL OPERATIONS (depends on operators and utils) ===
# partial operations (TODO partial transpose)
include("quantum/partial.jl")
export ptrace, ptrace_qubits, ptrace_2qubits
export ptrace_d, ptrace_qubits_d

# === QUANTUM CORRELATIONS (depends on partial operations) ===
# quantum correlations and entanglement
include("quantum/correlations.jl")
export concurrence, vn_entropy, mutual_info

# === LOCAL OPERATORS (depends on quantum utils) ===
# local operators used for hamiltonians
include("quantum/local.jl")
export LocalOperators

# === NETWORK THEORY (general - minimal dependencies) ===
# network theory functions
include("networks/networks.jl")
export get_link_weight, edges_to_adj, get_order
export degrees, laplacian

# === NETWORK METRICS (depends on network theory) ===
# network metrics functions
include("networks/metrics.jl")
export laplacian_spectrum, algebraic_connectivity

# === QUANTUM NETWORKS (depends on correlations and network theory) ===
# quantum networks functions
include("networks/q_networks.jl")
export node_entropies!, correlation_network
export correlation_edgelist, correlation_edgelist!

# === QRC HAMILTONIANS (depends on local operators) ===
# hamiltonians, unitaries, constructors
include("qrc/hamiltonians.jl")
export h_monroe, xx_monroe_obc, xx_monroe_pbc, z_noisy

# === QRC MEASUREMENTS (depends on operators and hamiltonians) ===
# quantum statistics and measurements for qrc
include("qrc/measurements.jl")
export local_measure, local_measure!
export montecarlo_measure, montecarlo_measure!
export simulated_measure, simulated_measure!
export get_binary_outcomes!

# === QRC TOOLS (depends on measurements and networks) ===
# simulations function for qelm for pure states
include("qrc/qrc_tools.jl")
export qelm_compute, qelm_compute_networks

# === MACHINE LEARNING (depends on flux and other) ===
# machine learning functions
export pca_analysis, rescale_data, accuracy
include("ml/functions.jl")

export train_epoch!, nn_layer
include("ml/linear.jl")

export ml_showvalues, @trainprogress
include("ml/progress.jl")

end # module QuantumReservoirComputing
