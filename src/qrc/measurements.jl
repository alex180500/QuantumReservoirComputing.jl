const QubitSpec{T<:Integer} = Union{Tuple{Vararg{T}},AbstractVector{T}}

function QubitSpec{T}(n::Integer) where {T<:Integer}
    return ntuple(i -> T(i - 1), n)
end

QubitSpec(n::T) where {T<:Integer} = QubitSpec{T}(n)

# local measurements for the full quantum state
function measure_local_func!(
    local_meas::AbstractVector{S},
    ρ::AbstractMatrix{T},
    meas::Function,
    qubits::QubitSpec,
    n_sys::Integer=get_nsys(ρ),
) where {T<:Number,S<:Real}
    @inbounds for i in eachindex(qubits)
        ρ_red = ptrace_qubits(ρ, qubits[i], n_sys)
        local_meas[i] = meas(ρ_red)::S
    end
    return local_meas
end

function measure_local_func!(
    local_meas::AbstractVector{S},
    ψ::AbstractVector{T},
    meas::Function,
    qubits::QubitSpec,
    n_sys::Integer=get_nsys(ψ),
) where {T<:Number,S<:Real}
    ρ = ψ * ψ'
    return measure_local_func!(local_meas, ρ, meas, qubits, n_sys)
end

function measure_local_func(
    ρ::AbstractMatrix{T}, meas::Function, qubits::QubitSpec, n_sys::Integer=get_nsys(ρ)
) where {T<:Number}
    local_meas = Vector{Float64}(undef, length(qubits))
    return measure_local_func!(local_meas, ρ, meas, qubits, n_sys)
end

function measure_local_func(
    ψ::AbstractVector{T}, meas::Function, qubits::QubitSpec, n_sys::Integer=get_nsys(ψ)
) where {T<:Number}
    local_meas = Vector{Float64}(undef, length(qubits))
    return measure_local_func!(local_meas, ψ, meas, qubits, n_sys)
end

function measure_local_func(
    ρ::AbstractMatrix{T}, meas::Function=avg_z, n_sys::Integer=get_nsys(ρ)
) where {T<:Number}
    return measure_local_func(ρ, meas, QubitSpec(n_sys), n_sys)
end

function measure_local_func(
    ψ::AbstractVector{T}, meas::Function=avg_z, n_sys::Integer=get_nsys(ψ)
) where {T<:Number}
    return measure_local_func(ψ, meas, QubitSpec(n_sys), n_sys)
end

# calculates the probabilities of measuring in the qubits basis
# simulating many measurements of a quantum state ρ
function montecarlo_measure_local(
    state::AbstractVecOrMat{T}, n_sys::Integer=get_nsys(ρ); n_samples::Integer=1_000_000
) where {T<:Complex}
    state_probs = get_probabilities(state)::Vector{real(T)}
    return montecarlo_measure_local(state_probs, n_sys; n_samples)
end

function montecarlo_measure_local(
    state_probs::AbstractVector{T},
    n_sys::Integer=get_nsys(state_probs);
    n_samples::Integer=1_000_000,
) where {T<:Real}
    outcomes = zeros(Float64, n_sys)
    rand_states = Vector{Int}(undef, n_samples)
    return montecarlo_measure_local!(outcomes, rand_states, state_probs)
end

function montecarlo_measure_local!(
    outcomes::AbstractVector{S},
    rand_states::AbstractVector{J},
    state_probs::AbstractVector{T},
) where {S<:AbstractFloat,J<:Integer,T<:Real}
    weight = Categorical(state_probs)
    rand!(weight, rand_states)
    counts = count_unique(rand_states)
    return get_binary_outcomes!(outcomes, counts, length(rand_states))
end

# MEASUREMENTS WITH MULTINOMIAL DISTRIBUTION

# infinite precision measurement just redirect to get_probabilities and get_probabilities!
measure(state::AbstractVecOrMat{T}) where {T<:Number} = get_probabilities(state)
function measure!(
    counts::AbstractVector{S}, state::AbstractVecOrMat{T}
) where {S<:Real,T<:Number}
    return get_probabilities!(counts, state)
end

# finite precision measurement using multinomial distribution
function measure(state_or_probs::AbstractVecOrMat{T}, n_samples::Integer) where {T<:Number}
    counts = Vector{Float64}(undef, size(state_or_probs, 1))
    return measure!(counts, state_or_probs, n_samples)
end

function measure!(
    counts::AbstractVector{S}, state::AbstractVecOrMat{T}, n_samples::Integer
) where {S<:Real,T<:Complex}
    state_probs = get_probabilities(state)::Vector{real(T)}
    return measure!(counts, state_probs, n_samples)
end

function measure!(
    counts::AbstractVector{S}, state_probs::AbstractVector{T}, n_samples::Integer
) where {S<:Real,T<:Real}
    distr = Multinomial(n_samples, state_probs)
    rand!(distr, counts)
    counts /= n_samples
    return counts
end

function measure_local(
    state_or_probs::AbstractVecOrMat{T}, n_samples::Integer
) where {T<:Number}
    outcomes = Vector{Float64}(undef, get_nsys(state_or_probs))
    counts = Vector{Float64}(undef, size(state_or_probs, 1))
    return measure_local!(outcomes, counts, state_or_probs, n_samples)
end

function measure_local!(
    outcomes::AbstractVector{S},
    counts::AbstractVector{S},
    state_or_probs::AbstractVecOrMat{T},
    n_samples::Integer,
) where {S<:Real,T<:Number}
    measure!(counts, state_or_probs, n_samples)
    return get_binary_outcomes!(outcomes, counts, 1.0)
end

# common function to get the binary outcomes

function get_binary_outcomes(
    weights::AbstractVector{T}, total_weight::T=sum(weights)
) where {T<:Real}
    outcomes = Vector{Float64}(undef, get_nsys(weights))
    return get_binary_outcomes!(outcomes, weights, total_weight)
end

function get_binary_outcomes!(
    outcomes::AbstractVector{S}, weights::AbstractVector{T}, total_weight::T=sum(weights)
) where {T<:Real,S<:Real}
    fill!(outcomes, zero(S))
    @inbounds for (idx, w) in enumerate(weights)
        state = idx - 1
        for j in eachindex(outcomes)
            outcomes[j] += w * get_bit(state, j)
        end
    end
    reverse!(outcomes)
    return outcomes / total_weight
end
