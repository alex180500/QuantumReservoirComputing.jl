function qelm_compute_networks(
    Λ::AbstractMatrix{T},
    input_states::AbstractMatrix{T},
    nq::Integer=get_nsys(input_states[:, 1]);
    n_states::Integer=size(input_states, 2)
) where {T<:Number}
    evol_ψ = Λ * input_states
    
    probs_t = similar(input_states, Float64)
    mi_net_t = Matrix{Float64}(undef, binomial(nq, 2), n_states)
    entropy_t = Matrix{Float64}(undef, nq, n_states)
    @inbounds @threads for i in 1:n_states
        ψ_t = view(evol_ψ, :, i)
        get_probabilities!(view(probs_t, :, i), ψ_t)
        ρ = ψ_t * ψ_t'
        correlation_edgelist!(view(mi_net_t, :, i), ρ, mutual_info, nq)
        node_entropies!(view(entropy_t, :, i), ρ, nq)
    end
    return (probs=probs_t, networks=mi_net_t, entropies=entropy_t)
end

function qelm_compute(
    Λ::AbstractMatrix{T},
    input_states::AbstractMatrix{T}
) where {T<:Number}
    evol_ψ = Λ * input_states
    
    probs_t = similar(input_states, Float64)
    @inbounds for i in axes(evol_ψ, 2)
        get_probabilities!(view(probs_t, :, i), view(evol_ψ, :, i))
    end
    return probs_t
end
