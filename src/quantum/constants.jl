# quantum states for qubits
const ket_0 = Vector{ComplexF64}([1, 0])
const ket_1 = Vector{ComplexF64}([0, 1])
const rho_0 = ket_0 * ket_0'
const rho_1 = ket_1 * ket_1'
const rho_mix = Matrix{ComplexF64}([1/2 0; 0 1/2])
const ket_p = (ket_0 + ket_1) / √2
const ket_m = (ket_0 - ket_1) / √2
const rho_p = ket_p * ket_p'
const rho_m = ket_m * ket_m'

# pauli matrices
const sig_x = Matrix{ComplexF64}([0 1; 1 0])
const sig_y = Matrix{ComplexF64}([0 -im; im 0])
const sig_z = Matrix{ComplexF64}([1 0; 0 -1])
const sig_p = Matrix{ComplexF64}([0 1; 0 0])
const sig_m = Matrix{ComplexF64}([0 0; 1 0])

# some other stuff
const eye2 = Matrix{ComplexF64}([1 0; 0 1])
const ⊗ = kron
