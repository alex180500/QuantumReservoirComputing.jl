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
    return get_edgelist!(edge_list, ρ, correlation, n_qubits)
end

function get_edgelist!(
    edge_list::AbstractVector{Float64},
    ρ::Matrix{T},
    correlation::Function,
    n_qubits::Int=get_nsys(ρ)
) where {T<:Number}
    count = 1
    @inbounds for i in 1:n_qubits, j in i+1:n_qubits
        reduced_ρ = ptrace_qubits(ρ, (i, j), n_qubits)::Matrix{ComplexF64}
        edge_list[count] = correlation(reduced_ρ)::Float64
        count += 1
    end
    return edge_list
end

# gets the link weight from the adjacency matrix
get_link_weight(network::Matrix{Float64}, edge::Tuple{Int,Int}) =
    network[edge...]

function edges_to_adj(edge_list::Vector{Float64}, n_qubits::Int)
    adj_mat = zeros(Float64, n_qubits, n_qubits)
    count = 1
    @inbounds for i in 1:n_qubits, j in i+1:n_qubits
        adj_mat[i, j] = adj_mat[j, i] = edge_list[count]
        count += 1
    end
    return adj_mat
end

# function get_mi_nets(list_ρs::Vector{Matrix{ComplexF64}}, n_sys::Int)
#     evol_nets = Matrix{Float64}(undef, binomial(n_sys, 2), length(list_ρs))
#     Threads.@threads for i in eachindex(list_ρs)
#         get_edgelist!(view(evol_nets, :, i), list_ρs[i], mutual_info, n_sys)
#     end
#     return evol_nets
# end

# function get_mi_nets(states::Matrix{ComplexF64}, n_sys::Int)
#     evol_nets = Matrix{Float64}(undef, binomial(n_sys, 2), size(states, 2))
#     Threads.@threads for i in axes(states, 2)
#         ρ = states[:, i] * states[:, i]'
#         get_edgelist!(view(evol_nets, :, i), ρ, mutual_info, n_sys)
#     end
#     return evol_nets
# end