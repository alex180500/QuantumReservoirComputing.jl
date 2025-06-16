module QuantumReservoirComputing

# Standard library
using LinearAlgebra: qr!, diag, Diagonal, eigvals, Hermitian, I, eigen
using Random: AbstractRNG, default_rng, rand!
using Base.Threads: @threads

# External packages
using Distributions: Categorical, Uniform, Multinomial
using MultivariateStats: PCA, fit, transform
using TensorOperations: tensortrace

# utils functions for bit operations
export get_bit, get_bit_table
include("utils/bits.jl")

# some generic utils functions
export get_mb, count_unique, eigvals_2
include("utils/generic.jl")

# network theory functions
export get_link_weight, edges_to_adj
include("networks/networks.jl")

# quantum networks functions
export node_entropies!, correlation_network
export correlation_edgelist, correlation_edgelist!
include("networks/q_networks.jl")

# constants useful for qubits, common operators
export ket_0, ket_1, ket_p, ket_m
export rho_0, rho_1, rho_mix, rho_p, rho_m
export sig_x, sig_y, sig_z, sig_p, sig_m
export eye2, ⊗, LocalOperators
include("quantum/constants.jl")

# quantum utils
export get_nsys, max_mixed, eye
export haar_unitary, haar_state, haar_dm
include("quantum/q_utils.jl")

# quantum correlations and entanglement
export concurrence, vn_entropy, mutual_info
include("quantum/correlations.jl")

# quantum operators functions
export avg_z, avg_x, avg_y, avg_zz
export local_ops, get_unitary
export get_probabilities, get_probabilities!
export get_bloch_vector
include("quantum/operators.jl")

# partial operations (TODO partial transpose)
export ptrace, ptrace_qubits, ptrace_2qubits
export ptrace_d, ptrace_qubits_d
include("quantum/partial.jl")

# hamiltonians, unitaries
export xx_monroe_obc, xx_monroe_pbc, z_noisy
export h_monroe
include("qrc/hamiltonians.jl")

# quantum statistics and measurements for qrc
export local_measure, local_measure!
export montecarlo_measure, montecarlo_measure!
export simulated_measure, simulated_measure!
export get_binary_outcomes!
include("qrc/measurements.jl")

# simulations function for qelm for pure states
export qelm_compute, qelm_compute_networks
include("qrc/qrc_tools.jl")

# machine learning functions
export pca_analysis, rescale_data
include("ml/functions.jl")

# TODO: FLUX SETUP

end # module QuantumReservoirComputing
