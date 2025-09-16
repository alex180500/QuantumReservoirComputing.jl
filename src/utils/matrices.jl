function directsum!(
    C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T<:Number}
    m, n = size(A)
    p, q = size(B)
    @inbounds begin
        fill!(view(C, 1:m, (n + 1):(n + q)), zero(T))
        fill!(view(C, (m + 1):(m + p), 1:n), zero(T))
        copyto!(view(C, 1:m, 1:n), A)
        copyto!(view(C, (m + 1):(m + p), (n + 1):(n + q)), B)
    end
    return C
end

function directsum(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Number}
    m, n = size(A)
    p, q = size(B)
    C = Matrix{T}(undef, m + p, n + q)
    return directsum!(C, A, B)
end

function directsum(a, b, c, xs...)
    return Base.afoldl(directsum, directsum(directsum(a, b), c), xs...)
end

const ⊕ = directsum

function kron_pow(A::AbstractVecOrMat{T}, N::Integer) where {T<:Number}
    result = A
    for _ in 2:N
        result = kron(result, A)
    end
    return result
end

const ⊗ = kron

function eigvals_2(mat::AbstractMatrix{T}) where {T<:Number}
    @inbounds begin
        a = mat[1, 1]
        b = mat[1, 2]
        d = mat[2, 2]
    end
    δ = a - d
    disc = sqrt(muladd(δ, δ, 4 * abs2(b)))
    t = (a + d)
    λ1 = (t - disc) / 2
    λ2 = (t + disc) / 2
    return Float64[λ1, λ2]
end
