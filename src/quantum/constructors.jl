# quantum states for qubits
const ket_0 = Vector{ComplexF64}([1, 0])
const ket_1 = Vector{ComplexF64}([0, 1])
const rho_0 = ket_0 * ket_0'
const rho_1 = ket_1 * ket_1'
const rho_mix = Matrix{ComplexF64}([1/2 0; 0 1/2])
const ket_p = (ket_0 + ket_1) / √2
const ket_m = (ket_0 - ket_1) / √2

# pauli matrices
const pauli_x = Matrix{ComplexF64}([0 1; 1 0])
const pauli_y = Matrix{ComplexF64}([0 -im; im 0])
const pauli_z = Matrix{ComplexF64}([1 0; 0 -1])
const pauli_p = Matrix{ComplexF64}([0 1; 0 0])
const pauli_m = Matrix{ComplexF64}([0 0; 1 0])

# gates
const H_gate = Matrix{ComplexF64}([1 1; 1 -1]) / √2
const T_gate = Matrix{ComplexF64}([1 0; 0 cispi(1 / 4)])
const S_gate = Matrix{ComplexF64}([1 0; 0 im])

# creates an identity matrix of dimension dim
const eye2 = Matrix{ComplexF64}([1 0; 0 1])
@memoize eye(dim::Integer) = Matrix{ComplexF64}(I, dim, dim)
@memoize eye_qubits(n_sys::Integer) = eye(2^n_sys)
# creates a maximally mixed state of dimension dim
@memoize max_mixed(dim::Integer) = eye(dim) / dim
