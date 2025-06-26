"""
    encode_qubit(θ::Float64, ϕ::Float64; scale_ϕ::Real=2)

Encode two classical data inputs `θ::Float64` and `ϕ::Float64` in [0, 1] into a quantum state `Vector{ComplexF64}`.

The encoding is done mapping `θ` into the polar angle [0, π] and `ϕ` into the azimuthal angle [0, scale_ϕ*π]. The resulting quantum qubit state is given by:
```math
\\ket{\\psi} = \\cos\\left(\\frac{\\theta}{2}\\right) \\ket{0} + \\sin\\left(\\frac{\\theta}{2}\right) \\exp{i\\phi} \\ket{1}
```
"""
function encode_qubit(θ::Real, ϕ::Real; scale_ϕ::Real=2)
    s, c = sincospi(θ / 2)
    return [c, s * cispi(scale_ϕ * ϕ)]
end

"""
    dense_angle_encoding(data_θ, data_ϕ::AbstractVector{T}; scale_ϕ::Real=2)

Encode 2N data inputs `data_θ::AbstractVector{T}` and `data_ϕ::AbstractVector{T}` into a quantum state `Vector{ComplexF64}` of N qubits. If `data_θ` and `data_ϕ` are vectors, the output is a single quantum state. If they are matrices, the output is a matrix of quantum states, where each column corresponds to a quantum state (uses data column-wise).

Based on 10.1103/PhysRevApplied.23.044024
"""
function dense_angle_encoding(
    data_θ::AbstractVector{T}, data_ϕ::AbstractVector{T}; scale_ϕ::Real=2
) where {T<:Real}
    return kron(encode_qubit.(data_θ, data_ϕ; scale_ϕ)...)
end
function dense_angle_encoding(
    data_θ::AbstractMatrix{T}, data_ϕ::AbstractMatrix{T}; scale_ϕ::Real=2
) where {T<:Real}
    n_qubits, n_states = size(data_θ)
    states = Matrix{ComplexF64}(undef, 2^n_qubits, n_states)
    for i in 1:n_states
        states[:, i] .= dense_angle_encoding(
            view(data_θ, :, i), view(data_ϕ, :, i); scale_ϕ=scale_ϕ
        )
    end
    return states
end
