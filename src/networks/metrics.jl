"""
    degrees(A::AbstractMatrix)

Computes the degree of each node in a graph from its weighted adjacency matrix `A`. It returns the sum of each row.
"""
function degrees(A::AbstractMatrix{T}) where {T<:Real}
    return vec(sum(A; dims=1))
end

"""
    degrees(edge_list::AbstractVector)

Method when the input is an edge list of weights of a complete graph in order (1-2, 1-3, ..., 2-3, ...).
"""
function degrees(edge_list::AbstractVector{T}) where {T<:Real}
    n = get_order(edge_list)
    deg = zeros(T, n)
    count = 1
    @inbounds for i in 1:n, j in (i + 1):n
        weight = edge_list[count]
        deg[i] += weight
        deg[j] += weight
        count += 1
    end
    return deg
end

"""
    laplacian(A::AbstractMatrix)

Computes the graph Laplacian matrix from an adjacency matrix `A`.
"""
function laplacian(A::AbstractMatrix{T}) where {T<:Real}
    D = Diagonal(degrees(A))
    return D - A
end

"""
    laplacian_spectrum(A::AbstractMatrix)

Returns the eigenvalues of the Laplacian of the adjacency matrix `A`.
"""
function laplacian_spectrum(A::AbstractMatrix{T}) where {T<:Real}
    return eigvals(laplacian(A))
end

"""
    algebraic_connectivity(A::AbstractMatrix[; normalized::Bool=false])

Computes the algebraic connectivity (second smallest Laplacian eigenvalue) of a graph defined by the adjacency matrix `A`. If `normalized` is true, it returns the normalized connectivity value divided by the maximum degree of the graph, otherwise it returns the raw algebraic connectivity value.
"""
function algebraic_connectivity(
    A::AbstractMatrix{T}; normalized::Bool=false
) where {T<:Real}
    λ = laplacian_spectrum(A)
    ac = λ[2]
    return normalized ? ac / maximum(degrees(A)) : ac
end

"""
    algebraic_connectivity(edge_list::AbstractVector[; normalized::Bool=false])

Method when the input is an edge list of weights of a complete graph in order (1-2, 1-3, ..., 2-3, ...).
"""
function algebraic_connectivity(
    edge_list::AbstractVector{T}; normalized::Bool=false
) where {T<:Real}
    A = edges_to_adj(edge_list, get_order(edge_list))
    return algebraic_connectivity(A; normalized=normalized)
end
