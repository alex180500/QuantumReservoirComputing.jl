# MEASUREMENTS WITH MULTINOMIAL DISTRIBUTION

# infinite precision measurement just redirect to get_probabilities and get_probabilities!
measure(state::AbstractVecOrMat{T}) where {T<:Number} = get_probabilities(state)

function measure!(
    counts::AbstractVector{S}, state::AbstractVecOrMat{T}
) where {S<:Real,T<:Number}
    return get_probabilities!(counts, state)
end

function measure_local(state::AbstractVecOrMat{T}) where {T<:Number}
    outcomes = Vector{Float64}(undef, get_nsys(state))
    return measure_local!(outcomes, state)
end

function measure_local!(
    outcomes::AbstractVector{S}, state::AbstractVecOrMat{T}
) where {S<:Real,T<:Number}
    state_probs = measure(state)::Vector{real(T)}
    return get_binary_outcomes!(outcomes, state_probs, 1.0)
end

# finite precision measurement using multinomial distribution
function measure(state::AbstractVecOrMat{T}, n_samples::Integer) where {T<:Number}
    counts = Vector{Int}(undef, size(state, 1))
    return measure!(counts, state, n_samples)
end

function measure!(
    counts::AbstractVector{S}, state::AbstractVecOrMat{T}, n_samples::Integer
) where {S<:Real,T<:Number}
    state_probs = get_probabilities(state)::Vector{real(T)}
    distr = Multinomial(n_samples, state_probs)
    return rand!(distr, counts)
end

# local version of measure (measure locally for each qubit)
function measure_local(state::AbstractVecOrMat{T}, n_samples::Integer) where {T<:Number}
    outcomes = Vector{Float64}(undef, get_nsys(state))
    counts = Vector{Int}(undef, size(state, 1))
    return measure_local!(outcomes, counts, state, n_samples)
end

function measure_local!(
    outcomes::AbstractVector{S},
    counts::AbstractVector{C},
    state::AbstractVecOrMat{T},
    n_samples::Integer,
) where {S<:Real,C<:Real,T<:Number}
    measure!(counts, state, n_samples)
    return get_binary_outcomes!(outcomes, counts, n_samples)
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

# QubitSpec, used for selecting which qubit to measure
const QubitSpec{T<:Integer} = Union{Tuple{Vararg{T}},AbstractVector{T}}

function QubitSpec{T}(n::Integer) where {T<:Integer}
    return ntuple(i -> T(i), n)
end
QubitSpec(n::T) where {T<:Integer} = QubitSpec{T}(n)
QubitSpec(::Type{T}, n::Integer) where {T<:Integer} = QubitSpec{T}(n)

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
    state::AbstractVecOrMat{T},
    meas::Function,
    qubits::QubitSpec,
    n_sys::Integer=get_nsys(state),
) where {T<:Number}
    local_meas = Vector{Float64}(undef, length(qubits))
    return measure_local_func!(local_meas, state, meas, qubits, n_sys)
end

function measure_local_func(
    state::AbstractVecOrMat{T}, meas::Function, n_sys::Integer=get_nsys(state)
) where {T<:Number}
    return measure_local_func(state, meas, QubitSpec(n_sys), n_sys)
end
