"""
    node_entropies!(entropies::AbstractVector, ρ::AbstractMatrix[, n_qubits::Integer=get_nsys(ρ)])

Computes the von Neumann entropy of each qubit in a quantum state. Fills the entropies vector with single-qubit reduced state entropies.
"""
function node_entropies!(
    entropies::AbstractVector{S}, ρ::AbstractMatrix{T}, n_qubits::Integer=get_nsys(ρ)
) where {S<:Real,T<:Number}
    @inbounds for i in 1:n_qubits
        reduced_ρ = ptrace_qubits(ρ, i, n_qubits)::Matrix{ComplexF64}
        entropies[i] = vn_entropy(reduced_ρ)::S
    end
    return entropies
end

"""
    node_entropies!(entropies::AbstractVector, ψ::AbstractVector[, n_qubits::Integer=get_nsys(ψ)])

Method for when the input is a pure state.
"""
function node_entropies!(
    entropies::AbstractVector{S}, ψ::AbstractVector{T}, n_qubits::Integer=get_nsys(ψ)
) where {S<:Real,T<:Number}
    return node_entropies!(entropies, ψ * ψ', n_qubits)
end

"""
    correlation_network(ρ::AbstractMatrix, correlation::Function[, n_qubits::Integer=get_nsys(ρ)])

Creates a correlation network adjacency matrix from a quantum state. (CITE 10.1103/PhysRevLett.119.225301)

See also [`correlation_edgelist`](@ref), [`correlation_edgelist!`](@ref).
"""
function correlation_network(
    ρ::AbstractMatrix{T}, correlation::Function, n_qubits::Integer=get_nsys(ρ)
) where {T<:Number}
    edge_list = correlation_edgelist(ρ, correlation, n_qubits)
    return edges_to_adj(edge_list, n_qubits)
end

"""
    correlation_edgelist(ρ::AbstractMatrix, correlation::Function[, n_qubits::Integer=get_nsys(ρ)])

Uses the given correlation function to compute pairwise qubit correlations. Returns a vector of pairwise correlation values between all qubit pairs corresponding to the complete graph weighted edgelist. (CITE 10.1103/PhysRevLett.119.225301)

See also [`correlation_network`](@ref), [`correlation_edgelist!`](@ref).
"""
function correlation_edgelist(
    ρ::AbstractMatrix{T}, correlation::Function, n_qubits::Integer=get_nsys(ρ)
) where {T<:Number}
    edge_list = Vector{Float64}(undef, binomial(n_qubits, 2))
    return correlation_edgelist!(edge_list, ρ, correlation, n_qubits)
end

"""
    correlation_edgelist!(edge_list::AbstractVector, ρ::AbstractMatrix, correlation::Function[, n_qubits::Integer=get_nsys(ρ)])

In-place version of `correlation_edgelist` that fills a pre-allocated edge list. (CITE 10.1103/PhysRevLett.119.225301)

See also [`correlation_network`](@ref), [`correlation_edgelist`](@ref).
"""
function correlation_edgelist!(
    edge_list::AbstractVector{S},
    ρ::AbstractMatrix{T},
    correlation::Function,
    n_qubits::Integer=get_nsys(ρ),
) where {S<:Real,T<:Number}
    count = 1
    @inbounds for i in 1:n_qubits, j in (i + 1):n_qubits
        reduced_ρ = ptrace_qubits(ρ, (i, j), n_qubits)::Matrix{ComplexF64}
        edge_list[count] = correlation(reduced_ρ)::S
        count += 1
    end
    return edge_list
end
