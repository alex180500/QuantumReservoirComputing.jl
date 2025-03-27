module QuantumReservoirComputing

using LinearAlgebra: qr!, diag, Diagonal, eigvals, Hermitian
using Random: AbstractRNG, default_rng
using TensorOperations: tensortrace
using Distributions: Categorical

export ket_0, ket_1, rho_0, rho_1, rho_mix
export sig_x, sig_y, sig_z, sig_p, sig_m, eye
export ⊗
include("constants.jl")

export get_mb, get_bit
include("generic_utils.jl")

export haar_unitary, haar_state, haar_dm
export concurrence, vn_entropy, mutual_info
export avg_z, avg_x, avg_y, avg_zz
include("quantum_functions.jl")

export ptrace
include("partial_operations.jl")

export local_ops
export avg_z_finite, measure_diagonal
include("qrc_utils.jl")

export get_network, get_link_weight
include("networks.jl")

end # module QuantumReservoirComputing
