# gets the link weight from the adjacency matrix
get_link_weight(network::Matrix{Float64}, edge::Tuple{Int,Int}) =
    network[edge...]

# creates the adjacency matrix from a quantum state
# given a correlation function
function get_network(
    ρ::Matrix{T},
    correlation::Function,
    n_qubits::Int=get_nsys(ρ)
) where {T<:Number}
    edge_list = get_edgelist(ρ, correlation, n_qubits)
    return edges_to_adj(edge_list, n_qubits)
end

# creates the edgelist from a quantum state
# given a correlation function
function get_edgelist(
    ρ::Matrix{T},
    correlation::Function,
    n_qubits::Int=get_nsys(ρ)
) where {T<:Number}
    edge_list = Vector{Float64}(undef, binomial(n_qubits, 2))
    count = 1
    @inbounds for i in 1:n_qubits, j in i+1:n_qubits
        reduced_ρ::Matrix{ComplexF64} = ptrace(ρ, (i, j), n_qubits)
        edge_list[count] = correlation(reduced_ρ)::Float64
        count += 1
    end
    return edge_list
end

function edges_to_adj(edge_list::Vector{Float64}, n_qubits::Int)
    adj_mat = zeros(Float64, n_qubits, n_qubits)
    count = 1
    @inbounds for i in 1:n_qubits, j in i+1:n_qubits
        adj_mat[i, j] = adj_mat[j, i] = edge_list[count]
        count += 1
    end
    return adj_mat
end

function get_mi_nets(list_ρs::Vector{Matrix{ComplexF64}}, n_sys::Int)
    evol_nets = Matrix{Float64}(undef, binomial(n_sys, 2), length(list_ρs))
    Threads.@threads for i in eachindex(list_ρs)
        evol_nets[:, i] = get_edgelist(list_ρs[i], mutual_info, n_sys)
    end
    return evol_nets
end