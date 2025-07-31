"""
    xx_monroe_pbc(σ_x::LocalOperators; α::Real)

Constructs XX interaction Hamiltonian with periodic boundary conditions and power-law decay interactions. Needs `σ_x` as [`LocalOperators`](@ref) of Pauli X operators. The long-range interactions, based on [ADD MONROE CITATION] are defined with coupling strength:
```math
J_{ij} = \\abs{i - j}^{-\\alpha}
```

See also [`xx_monroe_obc`](@ref) for open boundary conditions version.
"""
function xx_monroe_pbc(σ_x::LocalOperators{N,T}; α::Real) where {N,T<:Number}
    ham = zeros(ComplexF64, 2^N, 2^N)
    @inbounds for i in 1:N, j in (i + 1):N
        J_ij = min(abs(i - j), N - abs(i - j))^-α
        ham += J_ij * (σ_x[i] * σ_x[j])
    end
    return ham
end

"""
    xx_monroe_pbc(N::Integer; α::Real)

Method that creates [`LocalOperators`](@ref) of `N` Pauli X operators.
"""
function xx_monroe_pbc(N::Integer; α::Real)
    return xx_monroe_pbc(LocalOperators(pauli_x, N); α=α)
end

"""
    xx_monroe_obc(σ_x::LocalOperators; α::Real)

Open boundary conditions version of the XX interaction Hamiltonian in [`xx_monroe_pbc`](@ref).
"""
function xx_monroe_obc(σ_x::LocalOperators{N,T}; α::Real) where {N,T<:Number}
    ham = zeros(ComplexF64, 2^N, 2^N)
    @inbounds for i in 1:N, j in (i + 1):N
        J_ij = abs(i - j)^-α
        ham += J_ij * (σ_x[i] * σ_x[j])
    end
    return ham
end

"""
    xx_monroe_obc(N::Integer; α::Real)

Method that creates [`LocalOperators`](@ref) of `N` Pauli X operators.
"""
function xx_monroe_obc(N::Integer; α::Real)
    return xx_monroe_obc(LocalOperators(pauli_x, N); α=α)
end

"""
    z_noisy(σ_z::LocalOperators; W::Real, B::Real)

Local Z field hamiltonian with disorder. Needs `σ_z` as [`LocalOperators`](@ref) of Pauli Z operators. The local Z terms are:
```math
h_i = \\frac{D_i + B}{2}
```
where `D_i \\sim \\text{Unif}(-W, W)`.
"""
function z_noisy(σ_z::LocalOperators{N,T}; W::Real, B::Real) where {N,T<:Number}
    ham = zeros(ComplexF64, 2^N, 2^N)
    D = ifelse(W == 0, zeros(N), rand(Uniform(-W, W), N))
    @inbounds for i in 1:N
        ham += (D[i] + B) / 2 * σ_z[i]
    end
    return ham
end

"""
    z_noisy(N::Integer; W::Real, B::Real)

Method that creates [`LocalOperators`](@ref) of `N` Pauli Z operators.
"""
function z_noisy(N::Integer; W::Real, B::Real)
    return z_noisy(LocalOperators(pauli_z, N); W=W, B=B)
end

"""
    h_monroe(σ_x::LocalOperators, σ_z::LocalOperators; α::Real, W::Real, B::Real[, pbc::Bool=true])

Constructs a transverse field ising model (TIM) Hamiltonian with XX interaction and local disordered Z terms. Needs `σ_x` and `σ_z` as [`LocalOperators`](@ref) of Pauli X and Z operators, respectively. If `pbc` is true, periodic boundary conditions are used, otherwise open boundary conditions are applied. The Hamiltonian [ADD MONROE CITATION] is defined as:
```math
H = \\sum_{i, j} J_{ij} \\sigma^x_i \\sigma^x_j + \\sum_i \\frac{D_i + B}{2} \\sigma^z_i
```

See also [`xx_monroe_pbc`](@ref), [`xx_monroe_obc`](@ref), [`z_noisy`](@ref).
"""
function h_monroe(
    σ_x::LocalOperators{N,T},
    σ_z::LocalOperators{N,T};
    α::Real,
    W::Real,
    B::Real,
    pbc::Bool=true,
) where {N,T<:Number}
    if pbc
        H = xx_monroe_pbc(σ_x; α=α)
    else
        H = xx_monroe_obc(σ_x; α=α)
    end
    return H + z_noisy(σ_z; W=W, B=B)
end

"""
    h_monroe(N::Integer; α::Real, W::Real, B::Real[, pbc::Bool=true])

Method that creates [`LocalOperators`](@ref) of `N` Pauli X operators and `N` Pauli Z operators.
"""
function h_monroe(N::Integer; α::Real, W::Real, B::Real, pbc::Bool=true)
    σ_x = LocalOperators(pauli_x, N)
    σ_z = LocalOperators(pauli_z, N)
    return h_monroe(σ_x, σ_z; α=α, W=W, B=B, pbc=pbc)
end
