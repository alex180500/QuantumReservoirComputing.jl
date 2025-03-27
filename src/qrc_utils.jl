# operators, hamiltonians, unitaries

function local_ops(op::Matrix{T}, n_sys::Integer) where {T<:Number}
    dim = 2^n_sys
    all_ops = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_sys]
    @inbounds for idx in 1:n_sys
        op_list = fill(eye, n_sys)
        op_list[idx] = op
        all_ops[idx] = foldl(⊗, op_list)
    end
    return all_ops
end

# quantum statistics and measurements

function avg_z_finite(
    ρ::AbstractMatrix{T},
    trials::Integer=1_000_000,
    rng::AbstractRNG=default_rng()
) where {T<:Number}
    prob_up = real(ρ[1, 1])
    net_spin = count(rand(rng) < prob_up for _ in 1:trials)
    return 2 * net_spin / trials - 1
end

function measure_diagonal(
    ρ::Matrix{T},
    n_sys::Int=Int(log2(size(ρ, 1)));
    sample::Int=1_000_000,
) where {T<:Number}
    weights = Categorical(real(diag(ρ)))
    outcomes = zeros(Int, n_sys)
    @inbounds for _ in 1:sample
        state = rand(weights) - 1
        outcomes .+= get_bit.(state, 1:n_sys)
    end
    reverse!(outcomes)
    return @. 2 * outcomes / sample - 1
end
