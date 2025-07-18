"""
    get_link_weight(A::AbstractMatrix, edge::Union{Tuple,AbstractVector})

Gets the weight of a specific edge in a network adjacency matrix `A`.
"""
function get_link_weight(
    A::AbstractMatrix{T}, edge::Union{Tuple{J,J},AbstractVector{J}}
) where {T<:Real,J<:Integer}
    return A[edge[1], edge[2]]
end

"""
    edges_to_adj(edge_list::AbstractVector[, n_nodes::Integer=get_order(edge_list)])

Converts a weighted complete graph edge list to a symmetric adjacency matrix. The edges are assumed to be given in order such that the first node is connected to all others, the second node to all nodes after it, and so on.
"""
function edges_to_adj(
    edge_list::AbstractVector{T}, n_nodes::Integer=get_order(edge_list)
) where {T<:Real}
    adj_mat = zeros(T, n_nodes, n_nodes)
    count = 1
    @inbounds for i in 1:n_nodes, j in (i + 1):n_nodes
        adj_mat[i, j] = adj_mat[j, i] = edge_list[count]
        count += 1
    end
    return adj_mat
end

"""
    get_order(edge_list::AbstractVecOrMat)

Determines the number of nodes in a complete graph from the length of its edge list.
"""
function get_order(edge_list::AbstractVecOrMat{T}) where {T<:Real}
    n_edges = size(edge_list, 1)
    return Int((1 + sqrt(1 + 8 * n_edges)) / 2)
end
