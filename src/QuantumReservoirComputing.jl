module QuantumReservoirComputing

# Standard library imports
using LinearAlgebra:
    qr!, diag, Diagonal, eigvals, Hermitian, I, eigen, kron!, kron
using Random: AbstractRNG, default_rng, rand!
using Base.Threads: @threads

# External packages
using Distributions: Categorical, Uniform, Multinomial
using Memoization: @memoize
using MultivariateStats: PCA, fit, transform
using TensorOperations: tensortrace

# === UTILITIES (Foundation Layer - No Dependencies) ===
# utils functions for bit operations
export get_bit, get_bit_table
include("utils/bits.jl")

# some generic utils functions
export get_mb, count_unique, eigvals_2
include("utils/generic.jl")

# === QUANTUM CONSTANTS (Foundation Layer) ===
# constants useful for qubits, common operators
export ket_0, ket_1, ket_p, ket_m
export rho_0, rho_1, rho_mix, rho_p, rho_m
export sig_x, sig_y, sig_z, sig_p, sig_m
export eye2, ⊗
include("quantum/constants.jl")

# === QUANTUM UTILITIES (Depends on constants) ===
# quantum utils
export get_nsys, max_mixed, eye, eye_qubits
export haar_unitary, haar_state, haar_dm
include("quantum/q_utils.jl")

# === QUANTUM OPERATORS (Depends on constants and utils) ===
# quantum operators functions
export avg_z, avg_x, avg_y, avg_zz
export get_unitary, get_probabilities, get_probabilities!
export get_bloch_vector
include("quantum/operators.jl")

# === PARTIAL OPERATIONS (Depends on operators and utils) ===
# partial operations (TODO partial transpose)
export ptrace, ptrace_qubits, ptrace_2qubits
export ptrace_d, ptrace_qubits_d
include("quantum/partial.jl")

# === QUANTUM CORRELATIONS (Depends on partial operations) ===
# quantum correlations and entanglement
export concurrence, vn_entropy, mutual_info
include("quantum/correlations.jl")

# === LOCAL OPERATORS (Depends on quantum utils) ===
# local operators used for hamiltonians
export LocalOperators
include("quantum/local.jl")

# === NETWORK THEORY (General - Minimal Dependencies) ===
# network theory functions
export get_link_weight, edges_to_adj
include("networks/networks.jl")

# === QUANTUM NETWORKS (Depends on correlations and network theory) ===
# quantum networks functions
export node_entropies!, correlation_network
export correlation_edgelist, correlation_edgelist!
include("networks/q_networks.jl")

# === QRC HAMILTONIANS (Depends on local operators) ===
# hamiltonians, unitaries, constructors
export h_monroe, xx_monroe_obc, xx_monroe_pbc, z_noisy
include("qrc/hamiltonians.jl")

# === QRC MEASUREMENTS (Depends on operators and hamiltonians) ===
# quantum statistics and measurements for qrc
export local_measure, local_measure!
export montecarlo_measure, montecarlo_measure!
export simulated_measure, simulated_measure!
export get_binary_outcomes!
include("qrc/measurements.jl")

# === QRC TOOLS (High-level - Depends on measurements and networks) ===
# simulations function for qelm for pure states
export qelm_compute, qelm_compute_networks
include("qrc/qrc_tools.jl")

# === MACHINE LEARNING (Application Layer) ===
# machine learning functions
export pca_analysis, rescale_data
include("ml/functions.jl")

# TODO: FLUX SETUP

end # module QuantumReservoirComputing
