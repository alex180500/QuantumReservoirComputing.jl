function xx_monroe_pbc(σ_x::Vector{Matrix{ComplexF64}}, nq::Int; α::Float64)
    h_term = zeros(ComplexF64, 2^nq, 2^nq)
    @inbounds for i in 1:nq, j in i+1:nq
        J_ij = min(abs(i - j), nq - abs(i - j))^-α
        h_term += J_ij * (σ_x[i] * σ_x[j])
    end
    return h_term
end

function xx_monroe_obc(σ_x::Vector{Matrix{ComplexF64}}, nq::Int; α::Float64)
    h_term = zeros(ComplexF64, 2^nq, 2^nq)
    @inbounds for i in 1:nq, j in i+1:nq
        J_ij = abs(i - j)^-α
        h_term += J_ij * (σ_x[i] * σ_x[j])
    end
    return h_term
end

function z_noisy(
    σ_z::Vector{Matrix{ComplexF64}},
    nq::Int;
    W::Float64,
    B::Float64
)
    h_term = zeros(ComplexF64, 2^nq, 2^nq)
    rand_D = W == 0 ? zeros(nq) : rand(Uniform(-W, W), nq)
    @inbounds for i in 1:nq
        h_term += (rand_D[i] + B) / 2 * σ_z[i]
    end
    return h_term
end

function h_monroe(
    σ_x::Vector{Matrix{ComplexF64}},
    σ_z::Vector{Matrix{ComplexF64}},
    nq::Int;
    α::Float64,
    W::Float64,
    B::Float64,
    pbc::Bool=true
)
    if pbc
        H = xx_monroe_pbc(σ_x, nq, α=α)
    else
        H = xx_monroe_obc(σ_x, nq, α=α)
    end
    return H + z_noisy(σ_z, nq, W=W, B=B)
end
