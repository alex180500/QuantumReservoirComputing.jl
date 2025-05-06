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
