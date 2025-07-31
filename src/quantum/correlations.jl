# calculates the concurrence of a density matrix
function concurrence(ρ::AbstractMatrix{T}) where {T<:Number}
    Y = pauli_y ⊗ pauli_y
    R = ρ * (Y * conj(ρ) * Y)
    λ = sort(sqrt.(abs.(eigvals(R))); rev=true)
    return max(0.0, λ[1] - λ[2] - λ[3] - λ[4])
end

# calculates the von Neumann entropy of a density matrix
function vn_entropy(all_λ::AbstractVector{T}; base::Real=2, tol::Real=1e-15) where {T<:Real}
    S = zero(T)
    @inbounds for λ in all_λ
        if λ > tol
            S -= λ * log(base, λ)
        end
    end
    return S
end

function vn_entropy(ρ::AbstractMatrix{T}; base::Real=2, tol::Real=1e-15) where {T<:Number}
    all_λ = eigvals(Hermitian(ρ))
    return vn_entropy(all_λ; base=base, tol=tol)
end

# calculates the mutual information of a bipartite density matrix
function mutual_info(ρ::AbstractMatrix{T}; base::Real=2, tol::Real=1e-15) where {T<:Number}
    ρ_A = eigvals_2(ptrace_2qubits(ρ, 1))
    ρ_B = eigvals_2(ptrace_2qubits(ρ, 2))
    mi = vn_entropy(ρ_A; base, tol) + vn_entropy(ρ_B; base, tol) - vn_entropy(ρ; base, tol)
    return ifelse(mi < tol, zero(real(T)), mi)
end
