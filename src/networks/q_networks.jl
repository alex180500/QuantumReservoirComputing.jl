function node_entropies!(
    entropies::AbstractVector{S},
    ρ::AbstractMatrix{T},
    n_qubits::Integer=get_nsys(ρ)
) where {T<:Number,S<:Real}
    @inbounds for i in 1:n_qubits
        reduced_ρ = ptrace_qubits(ρ, i, n_qubits)::Matrix{ComplexF64}
        entropies[i] = vn_entropy(reduced_ρ)::S
    end
    return entropies
end

# creates the adjacency matrix from a quantum state
# given a correlation function
function correlation_network(
    ρ::AbstractMatrix{T},
    correlation::Function,
    n_qubits::Integer=get_nsys(ρ)
) where {T<:Number}
    edge_list = correlation_edgelist(ρ, correlation, n_qubits)
    return edges_to_adj(edge_list, n_qubits)
end

# creates the edgelist from a quantum state
# given a correlation function
function correlation_edgelist(
    ρ::AbstractMatrix{T},
    correlation::Function,
    n_qubits::Integer=get_nsys(ρ)
) where {T<:Number}
    edge_list = Vector{Float64}(undef, binomial(n_qubits, 2))
    return correlation_edgelist!(edge_list, ρ, correlation, n_qubits)
end

function correlation_edgelist!(
    edge_list::AbstractVector{S},
    ρ::AbstractMatrix{T},
    correlation::Function,
    n_qubits::Integer=get_nsys(ρ)
) where {T<:Number,S<:Real}
    count = 1
    @inbounds for i in 1:n_qubits, j in i+1:n_qubits
        reduced_ρ = ptrace_qubits(ρ, (i, j), n_qubits)::Matrix{ComplexF64}
        edge_list[count] = correlation(reduced_ρ)::S
        count += 1
    end
    return edge_list
end
