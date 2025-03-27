# networks and stuff

get_link_weight(network::Matrix{Float64}, edge::Tuple{Int,Int}) =
    network[edge...]

function get_network(
    ρ::Matrix{T},
    correlation::Function,
    n_qubits::Int=Int(log2(size(ρ, 1)))
) where {T<:Number}
    reduced_ρ = Matrix{ComplexF64}(undef, 4, 4)
    adj_mat = zeros(Float64, n_qubits, n_qubits)
    for i in 1:n_qubits, j in i+1:n_qubits
        reduced_ρ = ptrace(ρ, (i, j), n_qubits)
        corr_ij = correlation(reduced_ρ)
        adj_mat[i, j] = adj_mat[j, i] = corr_ij
    end
    return adj_mat
end
