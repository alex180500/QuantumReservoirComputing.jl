function laplacian_spectrum(A::AbstractMatrix{T}) where {T<:Real}
    return eigvals(laplacian(A))
end

function algebraic_connectivity(A::AbstractMatrix{T}) where {T<:Real}
    λ = laplacian_spectrum(A)
    return λ[2]
end
