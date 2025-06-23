# local measurements for the full quantum state
function local_measure!(
    local_meas::AbstractVector{S},
    ρ::AbstractMatrix{T},
    meas::Function,
    qubits::NTuple{N,Int},
    n_sys::Integer=get_nsys(ρ)
) where {N,T<:Number,S<:Real}
    @inbounds for i in eachindex(qubits)
        ρ_red = ptrace_qubits(ρ, qubits[i], n_sys)
        local_meas[i] = meas(ρ_red)::Float64
    end
    return local_meas
end

function local_measure(
    ρ::AbstractMatrix{T},
    meas::Function,
    qubits::NTuple{N,Int},
    n_sys::Integer=get_nsys(ρ)
) where {N,T<:Number}
    local_meas = Vector{Float64}(undef, N)
    return local_measure!(local_meas, ρ, meas, qubits, n_sys)
end

function local_measure(
    ρ::AbstractMatrix{T},
    meas::Function=avg_z,
    n_sys::Integer=get_nsys(ρ)
) where {T<:Number}
    return local_measure(ρ, meas, ntuple(i -> i, n_sys), n_sys)
end

# calculates the averages of all local z pauli matrices
# simulating many measurements of a quantum state ρ
function montecarlo_measure(
    ρ::AbstractMatrix{T},
    n_sys::Integer=get_nsys(ρ);
    n_samples::Integer=1_000_000
) where {T<:Number}
    ρ_populations = get_probabilities(ρ)::Vector{Float64}
    return montecarlo_measure(ρ_populations, n_sys; n_samples)
end

function montecarlo_measure(
    state_probs::AbstractVector{T},
    n_sys::Integer=get_nsys(state_probs);
    n_samples::Integer=1_000_000
) where {T<:Real}
    outcomes = zeros(Float64, n_sys)
    rand_states = Vector{Int}(undef, n_samples)
    return montecarlo_measure!(outcomes, rand_states, state_probs)
end

function montecarlo_measure!(
    outcomes::AbstractVector{S},
    rand_states::AbstractVector{I},
    state_probs::AbstractVector{T}
) where {S<:AbstractFloat,I<:Integer,T<:Real}
    weight = Categorical(state_probs)
    rand!(weight, rand_states)
    counts = count_unique(rand_states)
    return get_binary_outcomes!(outcomes, counts, length(rand_states))
end

# SIMULATED MEASUREMENTS

function simulated_measure(
    ρ::AbstractMatrix{T},
    n_samples::Integer=1_000_000
) where {T<:Number}
    state_probs = get_probabilities(ρ)
    return simulated_measure(state_probs, n_samples)
end

function simulated_measure(
    state_probs::AbstractVector{T},
    n_samples::Integer=1_000_000
) where {T<:Real}
    outcomes = zeros(Float64, length(state_probs))
    return simulated_measure!(outcomes, state_probs, n_samples)
end

function simulated_measure!(
    outcomes::AbstractVector{S},
    state_probs::AbstractVector{T},
    n_samples::Integer
) where {S<:AbstractFloat,T<:Real}
    counts = Vector{Int}(undef, length(state_probs))
    return simulated_measure!(outcomes, counts, state_probs, n_samples)
end

function simulated_measure!(
    outcomes::AbstractVector{S},
    counts::AbstractVector{I},
    state_probs::AbstractVector{T},
    n_samples::Integer
) where {S<:AbstractFloat,I<:Integer,T<:Real}
    distr = Multinomial(n_samples, state_probs)
    rand!(distr, counts)
    return get_binary_outcomes!(outcomes, counts, n_samples)
end

# common function to get the binary outcomes

function get_binary_outcomes!(
    outcomes::AbstractVector{S},
    weights::AbstractVector{T},
    total_weight::T=sum(weights)
) where {T<:Real,S<:AbstractFloat}
    fill!(outcomes, 0.0)
    @inbounds for (idx, w) in enumerate(weights)
        state = idx - 1
        for qubit in eachindex(outcomes)
            outcomes[qubit] += w * get_bit(state, qubit)
        end
    end
    reverse!(outcomes)
    @. outcomes = 1 - 2 * outcomes / total_weight
    return outcomes
end
