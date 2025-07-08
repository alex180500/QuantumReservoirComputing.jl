"""
    encode_qubit(θ::Real, ϕ::Real[; t::Real=2])

Encodes two classical values in ``[0, 1]`` into a single qubit quantum state using angle encoding. Maps `θ` to polar angle in ``[0, \\pi]`` and `ϕ` to azimuthal angle in ``[0, t\\pi]``, where `t = 2` by default, in the Bloch sphere representation.

Returns a quantum qubit of the form:
```math
\\ket{\\psi} = \\cos\\left(\\frac{\\theta}{2}\\right) \\ket{0} + \\sin\\left(\\frac{\\theta}{2}\right) \\exp{i\\phi} \\ket{1}
```
"""
function encode_qubit(θ::Real, ϕ::Real; t::Real=2)
    s, c = sincospi(θ / 2)
    return [c, s * cispi(t * ϕ)]
end

"""
    dense_angle_encoding(data_θ::AbstractVector, data_ϕ::AbstractVector; t::Real=2)

Encodes ``2N`` classical data points into ``N`` qubits using dense angle encoding. The final state is a tensor product of states encoded from each pair of `data_θ` and `data_ϕ` values.
"""
function dense_angle_encoding(
    data_θ::AbstractVector{T}, data_ϕ::AbstractVector{T}; t::Real=2
) where {T<:Real}
    return kron(encode_qubit.(data_θ, data_ϕ; t)...)
end

"""
    dense_angle_encoding(data_θ::AbstractMatrix, data_ϕ::AbstractMatrix; t::Real=2)

Method for encoding multiple states at once, each column of `data_θ` and `data_ϕ` will be encoded in one quantum state of ``N`` qubits. (CITE 10.1103/PhysRevApplied.23.044024)
"""
function dense_angle_encoding(
    data_θ::AbstractMatrix{T}, data_ϕ::AbstractMatrix{T}; t::Real=2
) where {T<:Real}
    n_qubits, n_states = size(data_θ)
    states = Matrix{ComplexF64}(undef, 2^n_qubits, n_states)
    for i in 1:n_states
        states[:, i] .= dense_angle_encoding(view(data_θ, :, i), view(data_ϕ, :, i); t)
    end
    return states
end
