# constants, qubits, density matrices, common operators

const ket_0 = Vector{ComplexF64}([1, 0])
const ket_1 = Vector{ComplexF64}([0, 1])
const rho_0 = ket_0 * ket_0'
const rho_1 = ket_1 * ket_1'
const rho_mix = Matrix{ComplexF64}([1/2 0; 0 1/2])

const sig_x = Matrix{ComplexF64}([0 1; 1 0])
const sig_y = Matrix{ComplexF64}([0 -im; im 0])
const sig_z = Matrix{ComplexF64}([1 0; 0 -1])
const sig_p = Matrix{ComplexF64}([0 1; 0 0])
const sig_m = Matrix{ComplexF64}([0 0; 1 0])

const eye = Matrix{ComplexF64}([1 0; 0 1])
const ⊗ = kron
