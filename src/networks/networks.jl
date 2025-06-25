function get_link_weight(
    network::AbstractMatrix{T},
    edge::Union{Tuple{I,I},AbstractVector{I}}
) where {T<:Real,I<:Integer}
    return network[edge[1], edge[2]]
end

function edges_to_adj(
    edge_list::AbstractVector{T},
    n_nodes::Integer=get_order(edge_list)
) where {T<:Real}
    adj_mat = zeros(T, n_nodes, n_nodes)
    count = 1
    @inbounds for i in 1:n_nodes, j in i+1:n_nodes
        adj_mat[i, j] = adj_mat[j, i] = edge_list[count]
        count += 1
    end
    return adj_mat
end

function laplacian(A::AbstractMatrix{T}) where {T<:Real}
    D = Diagonal(degrees(A))
    return D - A
end

function degrees(A::AbstractMatrix{T}) where {T<:Real}
    return vec(sum(A, dims=1))
end

function get_order(edge_list::AbstractVecOrMat{T}) where {T<:Real}
    n_edges = size(edge_list, 1)
    Int((1 + sqrt(1 + 8 * n_edges)) / 2)
end
