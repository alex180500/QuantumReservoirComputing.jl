function qelm_compute_networks(
    Λ::AbstractMatrix{T},
    input_states::AbstractMatrix{T},
    nq::Integer=get_nsys(input_states[:, 1]);
    n_states::Integer=size(input_states, 2)
) where {T<:Number}
    probs_t = Matrix{Float64}(undef, 2^nq, n_states)
    mi_net_t = Matrix{Float64}(undef, binomial(nq, 2), n_states)
    entropy_t = Matrix{Float64}(undef, nq, n_states)
    @threads for i in 1:n_states
        ψ_t = Λ * view(input_states, :, i)
        get_probabilities!(view(probs_t, :, i), ψ_t)
        ρ = ψ_t * ψ_t'
        correlation_edgelist!(view(mi_net_t, :, i), ρ, mutual_info, nq)
        node_entropies!(view(entropy_t, :, i), ρ, nq)
    end
    return probs_t, mi_net_t, entropy_t
end

function qelm_compute(
    Λ::AbstractMatrix{T},
    input_states::AbstractMatrix{T},
    nq::Integer=get_nsys(input_states[:, 1]);
    n_states::Integer=size(input_states, 2)
) where {T<:Number}
    probs_t = Matrix{Float64}(undef, 2^nq, n_states)
    @threads for i in 1:n_states
        ψ_t = Λ * view(input_states, :, i)
        get_probabilities!(view(probs_t, :, i), ψ_t)
    end
    return probs_t
end