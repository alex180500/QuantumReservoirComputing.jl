function get_link_weight(
    network::AbstractMatrix{T},
    edge::Union{Tuple{I,I},AbstractVector{I}}
) where {T<:Real,I<:Integer}
    return network[edge[1], edge[2]]
end

function edges_to_adj(
    edge_list::AbstractVector{T},
    n_nodes::Integer
) where {T<:Real}
    adj_mat = zeros(T, n_nodes, n_nodes)
    count = 1
    @inbounds for i in 1:n_nodes, j in i+1:n_nodes
        adj_mat[i, j] = adj_mat[j, i] = edge_list[count]
        count += 1
    end
    return adj_mat
end
