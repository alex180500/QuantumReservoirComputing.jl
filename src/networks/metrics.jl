function laplacian_spectrum(A::AbstractMatrix{T}) where {T<:Real}
    return eigvals(laplacian(A))
end

function algebraic_connectivity(
    A::AbstractMatrix{T};
    normalized::Bool=false
) where {T<:Real}
    λ = laplacian_spectrum(A)
    ac = λ[2]
    return normalized ? ac / maximum(degrees(A)) : ac
end
