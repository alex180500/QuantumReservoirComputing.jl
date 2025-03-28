# quantum utils functions and generics

function haar_unitary(n::Integer=2)
    Z = randn(ComplexF64, n, n)
    QR_decomp = qr!(Z)
    Λ = diag(QR_decomp.R)
    @. Λ /= abs(Λ)
    return QR_decomp.Q * Diagonal(Λ)
end

function haar_state(n::Integer=2)
    Z = randn(ComplexF64, n, n)
    QR_decomp = qr!(Z)
    return QR_decomp.Q[:, 1]
end

function haar_dm(n::Integer=2)
    ψ::Vector{ComplexF64} = haar_state(n)
    return ψ * ψ'
end

# entanglement, correlations, entropy

function concurrence(ρ::AbstractMatrix{T}) where {T<:Number}
    Y = sig_y ⊗ sig_y
    R = ρ * (Y * conj(ρ) * Y)
    λ = sort(sqrt.(abs.(eigvals(R))), rev=true)
    return max(0.0, λ[1] - λ[2] - λ[3] - λ[4])
end

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

function mutual_info(ρ::AbstractMatrix{T}) where {T<:Number}
    ρ_A = ptrace(ρ, 1)
    ρ_B = ptrace(ρ, 2)
    return vn_entropy(ρ_A) + vn_entropy(ρ_B) - vn_entropy(ρ)
end

# fast average values of pauli operators

avg_z(ρ::Matrix{<:Number}) = real(ρ[1, 1] - ρ[2, 2])
avg_x(ρ::Matrix{<:Number}) = real(ρ[1, 2] + ρ[2, 1])
avg_y(ρ::Matrix{<:Number}) = real(im * (ρ[1, 2] - ρ[2, 1]))
avg_zz(ρ::Matrix{<:Number}) = real(ρ[1, 1] - ρ[2, 2] - ρ[3, 3] + ρ[4, 4])
