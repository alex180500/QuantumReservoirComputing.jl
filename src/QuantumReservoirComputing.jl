module QuantumReservoirComputing

using LinearAlgebra: qr!, diag, Diagonal, eigvals, Hermitian, I
using Random: AbstractRNG, default_rng, rand!
using TensorOperations: tensortrace
using Distributions: Categorical, Uniform

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
export get_network, get_link_weight
include("networks.jl")

# quantum utils, haar random
# entanglement, entropy, correlations
# fast average values of pauli operators
export max_mixed, eye
export haar_unitary, haar_state, haar_dm
export concurrence, vn_entropy, mutual_info
export avg_z, avg_x, avg_y, avg_zz, avg_z_finite
include("quantum_functions.jl")

# partial operations, for now partial trace
# TODO ptranspose
export ptrace, ptrace_diag
include("partial_operations.jl")

# operators, hamiltonians, unitaries and other utils for qrc
# quantum statistics and measurements
export local_ops, local_measure, quantum_measure, quantum_measure!
export h_monroe_obc, h_monroe_pbc
include("qrc_utils.jl")

end # module QuantumReservoirComputing
