# given an operator op construct a list of local operators
# for each qubit in the system
function local_ops(op::Matrix{T}, n_sys::Integer) where {T<:Number}
    all_ops = [Matrix{ComplexF64}(undef, 2^n_sys, 2^n_sys) for _ in 1:n_sys]
    all_eyes = [eye(2^i) for i in 1:n_sys-1]
    all_ops[1] = op ⊗ all_eyes[n_sys-1]
    @inbounds for idx in 2:n_sys-1
        all_ops[idx] = all_eyes[idx-1] ⊗ op ⊗ all_eyes[n_sys-idx]
    end
    all_ops[n_sys] = all_eyes[n_sys-1] ⊗ op
    return all_ops
end

# local measurements for the full quantum state
# when i want to measure all subsystems
function local_measure(
    ρ::AbstractMatrix{T},
    meas::Function,
    n_sys::Int=get_nsys(ρ)
) where {T<:Number}
    local_measure(ρ, meas, ntuple(i -> i, n_sys), n_sys)
end

# local measurements for the full quantum state
function local_measure(
    ρ::AbstractMatrix{T},
    meas::Function,
    qubits::NTuple{N,Int},
    n_sys::Int=get_nsys(ρ)
) where {N,T<:Number}
    local_meas = Vector{Float64}(undef, N)
    @inbounds for (idx, q) in enumerate(qubits)
        ρ_red = ptrace(ρ, q, n_sys, d=2)
        local_meas[idx] = meas(ρ_red)
    end
    return local_meas
end

# calculates the averages of all local z pauli matrices
# simulating many measurements of a quantum state ρ
function quantum_measure(
    ρ::AbstractMatrix{T},
    n_sys::Int=get_nsys(ρ);
    sample::Int=1_000_000
) where {T<:Number}
    rand_states = Vector{Int}(undef, sample)
    return quantum_measure!(rand_states, ρ, n_sys)
end

function quantum_measure!(
    rand_states::AbstractVector{Int},
    ρ::AbstractMatrix{T},
    n_sys::Int=get_nsys(ρ)
) where {T<:Number}
    weight = Categorical(real(diag(ρ)))
    rand!(weight, rand_states)
    return get_binary_outcomes(rand_states, n_sys)
end

function quantum_measure!(
    rand_states::AbstractVector{Int},
    state_probs::AbstractVector{Float64},
    n_sys::Int=length(state_probs)
)
    weight = Categorical(state_probs)
    rand!(weight, rand_states)
    return get_binary_outcomes(rand_states, n_sys)
end

function get_binary_outcomes(
    rand_states::AbstractVector{Int},
    n_sys::Int
)
    outcomes = zeros(Float64, n_sys)
    @inbounds @simd for state in rand_states
        outcomes .+= get_bit.(state - 1, 1:n_sys)
    end
    reverse!(outcomes)
    return 2 .* outcomes ./ length(rand_states) .- 1
end

# interesting hamiltonians
function h_monroe_obc(
    σ_x::Vector{Matrix{ComplexF64}},
    σ_z::Vector{Matrix{ComplexF64}},
    nq::Int;
    α::Float64,
    W::Float64,
    B::Float64
)
    H = zeros(ComplexF64, 2^nq, 2^nq)
    rand_D = W == 0 ? zeros(nq) : rand(Uniform(-W, W), nq)
    @inbounds for i in 1:nq
        H += (rand_D[i] + B) / 2 * σ_z[i]
    end
    @inbounds for i in 1:nq, j in i+1:nq
        J_ij = abs(i - j)^-α
        H += J_ij * (σ_x[i] * σ_x[j])
    end
    return H
end

# interesting hamiltonians
function h_monroe_pbc(
    σ_x::Vector{Matrix{ComplexF64}},
    σ_z::Vector{Matrix{ComplexF64}},
    nq::Int;
    α::Float64,
    W::Float64,
    B::Float64
)
    H = zeros(ComplexF64, 2^nq, 2^nq)
    rand_D = W == 0 ? zeros(nq) : rand(Uniform(-W, W), nq)
    @inbounds for i in 1:nq
        H += (rand_D[i] + B) / 2 * σ_z[i]
    end
    @inbounds for i in 1:nq, j in i+1:nq
        J_ij = min(abs(i - j), nq - abs(i - j))^-α
        H += J_ij * (σ_x[i] * σ_x[j])
    end
    return H
end
