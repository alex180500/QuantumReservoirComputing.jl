# fast average values of pauli matrices of a qubit (or 2)
avg_z(ρ::AbstractMatrix{<:Number}) = real(ρ[1, 1] - ρ[2, 2])
avg_z(ψ::AbstractVector{<:Complex}) = abs2(ψ[1]) - abs2(ψ[2])
avg_z(pops::AbstractVector{<:Real}) = pops[1] - pops[2]
avg_x(ρ::AbstractMatrix{<:Number}) = real(ρ[1, 2] + ρ[2, 1])
avg_x(ψ::AbstractVector{<:Complex}) = 2 * real(conj(ψ[1]) * ψ[2])
avg_y(ρ::AbstractMatrix{<:Number}) = real(im * (ρ[1, 2] - ρ[2, 1]))
avg_y(ψ::AbstractVector{<:Complex}) = 2 * imag(conj(ψ[2]) * ψ[1])
avg_zz(ρ::AbstractMatrix{<:Number}) = real(ρ[1, 1] - ρ[2, 2] - ρ[3, 3] + ρ[4, 4])
avg_zz(ψ::AbstractVector{<:Complex}) = abs2(ψ[1]) - abs2(ψ[2]) - abs2(ψ[3]) + abs2(ψ[4])

# creates a unitary matrix from the hamiltonian
function get_unitary(H::AbstractMatrix{T}, δt::Real) where {T<:Number}
    # h_eigvals, h_eigvecs = eigen(H)
    # return h_eigvecs * exp(Diagonal(-im * δt .* h_eigvals)) * h_eigvecs'
    return cis(-δt * H)
end

function get_probabilities(ρ::AbstractMatrix{T}) where {T<:Number}
    prob_vec = Vector{real(T)}(undef, size(ρ, 1))
    return get_probabilities!(prob_vec, ρ)
end

function get_probabilities(ψ::AbstractVector{T}) where {T<:Number}
    prob_vec = Vector{real(T)}(undef, length(ψ))
    return get_probabilities!(prob_vec, ψ)
end

function get_probabilities!(
    prob_vec::AbstractVector{S}, ψ::AbstractVector{T}
) where {T<:Number,S<:Real}
    @inbounds for i in eachindex(prob_vec)
        prob_vec[i] = abs2(ψ[i])
    end
    return prob_vec
end

function get_probabilities!(
    prob_vec::AbstractVector{S}, ρ::AbstractMatrix{T}
) where {T<:Number,S<:Real}
    @inbounds for i in eachindex(prob_vec)
        prob_vec[i] = real(ρ[i, i])
    end
    return prob_vec
end

function get_bloch_vector(ψ::AbstractVector{T}) where {T<:Number}
    return get_bloch_vector(ψ * ψ')
end

function get_bloch_vector(ρ::AbstractMatrix{T}) where {T<:Number}
    return [2 * real(ρ[1, 2]), 2 * imag(ρ[2, 1]), real(ρ[1, 1] - ρ[2, 2])]
end
