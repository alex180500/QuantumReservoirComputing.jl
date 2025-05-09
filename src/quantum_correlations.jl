# calculates the concurrence of a density matrix
function concurrence(ρ::AbstractMatrix{T}) where {T<:Number}
    Y = sig_y ⊗ sig_y
    R = ρ * (Y * conj(ρ) * Y)
    λ = sort(sqrt.(abs.(eigvals(R))), rev=true)
    return max(0.0, λ[1] - λ[2] - λ[3] - λ[4])
end

# calculates the von Neumann entropy of a density matrix
function vn_entropy(
    ρ::AbstractMatrix{T};
    tol::Float64=eps(real(T))
) where {T<:Number}
    all_λ = eigvals(Hermitian(ρ))
    S = 0.0
    for λ in all_λ
        if λ > tol
            S -= λ * log2(λ)
        end
    end
    return S
end

# calculates the mutual information of a bipartite density matrix
function mutual_info(ρ::AbstractMatrix{T}) where {T<:Number}
    ρ_A = ptrace(ρ, 1)
    ρ_B = ptrace(ρ, 2)
    return vn_entropy(ρ_A) + vn_entropy(ρ_B) - vn_entropy(ρ)
end
