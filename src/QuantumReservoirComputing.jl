module QuantumReservoirComputing

using LinearAlgebra: qr!, diag, Diagonal, eigvals, Hermitian, I
using Random: AbstractRNG, default_rng
using TensorOperations: tensortrace
using Distributions: Categorical, Uniform

# constants useful for qubits, common operators
export ket_0, ket_1, ket_p, ket_m
export rho_0, rho_1, rho_mix, rho_p, rho_m
export sig_x, sig_y, sig_z, sig_p, sig_m
export eye2, ⊗
include("constants.jl")

# utils functions
export get_mb, get_bit, get_nqubits
include("generic_utils.jl")

# network theory functions
export get_network, get_link_weight
include("networks.jl")

# quantum utils, haar random
# entanglement, entropy, correlations
# fast average values of pauli operators
export max_mixed, eye
export haar_unitary, haar_state, haar_dm
export concurrence, vn_entropy, mutual_info
export avg_z, avg_x, avg_y, avg_zz
include("quantum_functions.jl")

# partial operations, for now partial trace
# TODO ptranspose
export ptrace
include("partial_operations.jl")

# operators, hamiltonians, unitaries and other utils for qrc
# quantum statistics and measurements
export local_ops
export avg_z_finite, measure_diagonal, local_measure
export h_monroe_obc, h_monroe_pbc
include("qrc_utils.jl")

end # module QuantumReservoirComputing
