module QuantumReservoirComputing

using Distributions: Categorical, Uniform, Multinomial
using LinearAlgebra:
    qr!, diag, Diagonal, eigvals, Hermitian, I, eigen
using MultivariateStats: PCA, fit, transform
using Random: AbstractRNG, default_rng, rand!
using TensorOperations: tensortrace

# utils functions
export get_mb, count_unique, eigvals_2x2
include("utils/generic.jl")
export get_bit, get_bit_table
include("utils/bits.jl")

# network theory functions
export get_link_weight, edges_to_adj
export get_network, get_edgelist, get_edgelist!
include("networks/networks.jl")

# constants useful for qubits, common operators
export ket_0, ket_1, ket_p, ket_m
export rho_0, rho_1, rho_mix, rho_p, rho_m
export sig_x, sig_y, sig_z, sig_p, sig_m
export eye2, ⊗
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
include("quantum/operators.jl")

# partial operations (TODO partial transpose)
export ptrace, ptrace_qubits, ptrace_2qubits
export ptrace_d, ptrace_qubits_d
include("quantum/partial.jl")

# hamiltonians, unitaries
export xx_monroe_obc, xx_monroe_pbc, z_noisy
export h_monroe
include("qrc_hamiltonians.jl")

# quantum statistics and measurements for qrc
export local_measure, local_measure!, local_measure_d, local_measure_d!
export quantum_measure, quantum_measure!
export simulated_measure, simulated_measure!
export get_binary_outcomes!
include("qrc_measurements.jl")

# machine learning functions
export pca_analysis
include("ml_functions.jl")

end # module QuantumReservoirComputing
