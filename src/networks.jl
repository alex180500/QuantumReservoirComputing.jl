# gets the link weight from the adjacency matrix
get_link_weight(network::Matrix{Float64}, edge::Tuple{Int,Int}) =
    network[edge...]

# creates the adjacency matrix from a quantum state
# given a correlation function
function get_network(
    ρ::Matrix{T},
    correlation::Function,
    n_qubits::Int=get_nqubits(ρ)
) where {T<:Number}
    reduced_ρ = Matrix{ComplexF64}(undef, 4, 4)
    adj_mat = zeros(Float64, n_qubits, n_qubits)
    for i in 1:n_qubits, j in i+1:n_qubits
        reduced_ρ::Matrix{ComplexF64} = ptrace(ρ, (i, j), n_qubits)
        corr_ij::Float64 = correlation(reduced_ρ)
        adj_mat[i, j] = adj_mat[j, i] = corr_ij
    end
    return adj_mat
end
