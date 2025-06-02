# local measurements for the full quantum state

function local_measure!(
    local_meas::AbstractVector{Float64},
    ρ::AbstractMatrix{T},
    meas::F,
    qubits::NTuple{N,Int},
    n_sys::Int=get_nsys(ρ)
) where {N,T<:Number,F<:Function}
    @inbounds for i in eachindex(qubits)
        ρ_red = ptrace_qubits(ρ, qubits[i], n_sys)
        local_meas[i] = meas(ρ_red)::Float64
    end
    return local_meas
end

function local_measure(
    ρ::AbstractMatrix{T},
    meas::F,
    qubits::NTuple{N,Int},
    n_sys::Int=get_nsys(ρ)
) where {N,T<:Number,F<:Function}
    local_meas = Vector{Float64}(undef, N)
    local_measure!(local_meas, ρ, meas, qubits, n_sys)
    return local_meas
end

function local_measure(
    ρ::AbstractMatrix{T},
    meas::F=avg_z,
    n_sys::Int=get_nsys(ρ)
) where {T<:Number,F<:Function}
    local_measure(ρ, meas, ntuple(i -> i, n_sys), n_sys)
end

# local measuremenet but only use the diagonal part

function local_measure_d!(
    local_meas::AbstractVector{Float64},
    ρ::AbstractMatrix{T},
    meas::F,
    qubits::NTuple{N,Int},
    n_sys::Int=get_nsys(ρ)
) where {N,T<:Number,F<:Function}
    @inbounds for i in eachindex(qubits)
        ρ_red = ptrace_qubits_d(ρ, qubits[i], n_sys)
        local_meas[i] = meas(ρ_red)::Float64
    end
    return local_meas
end

function local_measure_d(
    ρ::AbstractMatrix{T},
    meas::F,
    qubits::NTuple{N,Int},
    n_sys::Int=get_nsys(ρ)
) where {N,T<:Number,F<:Function}
    local_meas = Vector{Float64}(undef, N)
    local_measure_d!(local_meas, ρ, meas, qubits, n_sys)
    return local_meas
end

function local_measure_d(
    ρ::AbstractMatrix{T},
    meas::F,
    n_sys::Int=get_nsys(ρ)
) where {T<:Number,F<:Function}
    local_measure_d(ρ, meas, ntuple(i -> i, n_sys), n_sys)
end

# calculates the averages of all local z pauli matrices
# simulating many measurements of a quantum state ρ
function quantum_measure(
    ρ::AbstractMatrix{T},
    n_sys::Int=get_nsys(ρ);
    sample::Int=1_000_000
) where {T<:Number}
    ρ_populations::Vector{Float64} = real(diag(ρ))
    return quantum_measure(ρ_populations, n_sys, sample=sample)
end

function quantum_measure(
    state_probs::AbstractVector{Float64},
    n_sys::Int=get_nsys(state_probs);
    n_samples::Int=1_000_000
)
    outcomes = zeros(Float64, n_sys)
    rand_states = Vector{Int}(undef, n_samples)
    return quantum_measure!(outcomes, rand_states, state_probs)
end

function quantum_measure!(
    outcomes::AbstractVector{Float64},
    rand_states::AbstractVector{Int},
    state_probs::AbstractVector{Float64}
)
    weight = Categorical(state_probs)
    rand!(weight, rand_states)
    counts = count_unique(rand_states, length(state_probs))
    return get_binary_outcomes!(outcomes, counts, length(rand_states))
end

# SIMULATED MEASUREMENTS

function simulated_measure(
    ρ::AbstractMatrix{T},
    n_samples::Int=1_000_000
) where {T<:Number}
    state_probs = real(diag(ρ))
    return simulated_measure(state_probs, n_samples)
end

function simulated_measure(
    state_probs::AbstractVector{Float64},
    n_samples::Int=1_000_000
)
    outcomes = zeros(Float64, length(state_probs))
    return simulated_measure!(outcomes, state_probs, n_samples)
end

function simulated_measure!(
    outcomes::AbstractVector{Float64},
    state_probs::AbstractVector{Float64},
    n_samples::Int
)
    counts = Vector{Int}(undef, length(state_probs))
    return simulated_measure!(outcomes, state_probs, counts, n_samples)
end

function simulated_measure!(
    outcomes::AbstractVector{Float64},
    state_probs::AbstractVector{Float64},
    counts::AbstractVector{Int},
    n_samples::Int
)
    distr = Multinomial(n_samples, state_probs)
    rand!(distr, counts)
    return get_binary_outcomes!(outcomes, counts, n_samples)
end

# common function to get the binary outcomes

function get_binary_outcomes!(
    outcomes::AbstractVector{Float64},
    counts::AbstractVector{Int},
    n_samples::Int=sum(counts)
)
    fill!(outcomes, 0.0)
    @inbounds for (idx, c) in enumerate(counts)
        state = idx - 1
        for qubit in eachindex(outcomes)
            outcomes[qubit] += c * get_bit(state, qubit)
        end
    end
    reverse!(outcomes)
    @. outcomes = 1 - 2 * outcomes / n_samples
    return outcomes
end