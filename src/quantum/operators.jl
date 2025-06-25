# fast average values of pauli matrices of a qubit (or 2)
avg_z(ρ::AbstractMatrix{<:Number}) = real(ρ[1, 1] - ρ[2, 2])
avg_z(ρ_population::AbstractVector{<:Real}) = ρ_population[1] - ρ_population[2]
avg_x(ρ::AbstractMatrix{<:Number}) = real(ρ[1, 2] + ρ[2, 1])
avg_y(ρ::AbstractMatrix{<:Number}) = real(im * (ρ[1, 2] - ρ[2, 1]))
avg_zz(ρ::AbstractMatrix{<:Number}) = real(ρ[1, 1] - ρ[2, 2] - ρ[3, 3] + ρ[4, 4])

# creates a unitary matrix from the hamiltonian
function get_unitary(H::AbstractMatrix{T}, δt::Real) where {T<:Number}
    # h_eigvals, h_eigvecs = eigen(H)
    # return h_eigvecs * exp(Diagonal(-im * δt .* h_eigvals)) * h_eigvecs'
    return cis(-δt * H)
end

function get_probabilities(ψ::AbstractVector{T}) where {T<:Number}
    probs = Vector{Float64}(undef, length(ψ))
    return get_probabilities!(probs, ψ)
end

function get_probabilities!(
    probs::AbstractVector{S}, ψ::AbstractVector{T}
) where {T<:Number,S<:Real}
    @inbounds for i in eachindex(probs)
        probs[i] = abs2(ψ[i])
    end
    return probs
end

function get_probabilities(ρ::AbstractMatrix{T}) where {T<:Number}
    probs = Vector{Float64}(undef, size(ρ, 1))
    return get_probabilities!(probs, ρ)
end

function get_probabilities!(
    probs::AbstractVector{S}, ρ::AbstractMatrix{T}
) where {T<:Number,S<:Real}
    @inbounds for i in eachindex(probs)
        probs[i] = real(ρ[i, i])
    end
    return probs
end

function get_bloch_vector(ψ::AbstractVector{T}) where {T<:Number}
    return get_bloch_vector(ψ * ψ')
end

function get_bloch_vector(ρ::AbstractMatrix{T}) where {T<:Number}
    return [2 * real(ρ[1, 2]), 2 * imag(ρ[2, 1]), real(ρ[1, 1] - ρ[2, 2])]
end
