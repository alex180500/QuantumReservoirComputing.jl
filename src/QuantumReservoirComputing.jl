module QuantumReservoirComputing

using LinearAlgebra: qr!, diag, Diagonal, eigvals, Hermitian, I
using Random: AbstractRNG, default_rng, rand!
using Distributions: Categorical, Uniform
using TensorOperations: tensortrace

# constants useful for qubits, common operators
export ket_0, ket_1, ket_p, ket_m
export rho_0, rho_1, rho_mix, rho_p, rho_m
export sig_x, sig_y, sig_z, sig_p, sig_m
export eye2, ⊗
include("constants.jl")

# utils functions
export get_mb, get_bit, get_nsys
include("generic_utils.jl")

# network theory functions
export get_network, get_link_weight, get_edgelist
include("networks.jl")

# quantum utils, haar random
# entanglement, entropy, correlations
# fast average values of pauli operators
export max_mixed, eye
export haar_unitary, haar_state, haar_dm
include("quantum_utils.jl")

export avg_z, avg_x, avg_y, avg_zz, avg_z_finite
include("quantum_calculations.jl")

export concurrence, vn_entropy, mutual_info
include("quantum_correlations.jl")

# partial operations, for now partial trace
# TODO ptranspose
export ptrace, ptrace_diag
include("partial_operations.jl")

# operators, hamiltonians, unitaries and other utils for qrc
# quantum statistics and measurements
export local_ops, local_measure
export quantum_measure, quantum_measure!, get_binary_outcomes
export h_monroe_obc, h_monroe_pbc
include("qrc_utils.jl")
include("qrc_hamiltonians.jl")

end # module QuantumReservoirComputing
