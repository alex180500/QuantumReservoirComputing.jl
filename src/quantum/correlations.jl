# calculates the concurrence of a density matrix
function concurrence(ρ::AbstractMatrix{T}) where {T<:Number}
    Y = sig_y ⊗ sig_y
    R = ρ * (Y * conj(ρ) * Y)
    λ = sort(sqrt.(abs.(eigvals(R))), rev=true)
    return max(0.0, λ[1] - λ[2] - λ[3] - λ[4])
end

# calculates the von Neumann entropy of a density matrix
function vn_entropy(
    all_λ::AbstractVector{T};
    tol::Float64=eps(real(T))
) where {T<:Number}
    S = 0.0
    @inbounds for λ in all_λ
        if λ > tol
            S -= λ * log2(λ)
        end
    end
    return S
end

function vn_entropy(
    ρ::AbstractMatrix{T};
    tol::Float64=eps(real(T))
) where {T<:Number}
    all_λ = eigvals(Hermitian(ρ))
    return vn_entropy(all_λ, tol=tol)
end

# calculates the mutual information of a bipartite density matrix
function mutual_info(ρ::AbstractMatrix{T}; tol=eps(real(T))) where {T<:Number}
    ρ_A = eigvals_2x2(ptrace_2qubits(ρ, 1))
    ρ_B = eigvals_2x2(ptrace_2qubits(ρ, 2))
    return vn_entropy(ρ_A; tol) + vn_entropy(ρ_B; tol) - vn_entropy(ρ; tol)
end
