"""
    laplacian_spectrum(A::AbstractMatrix)

Returns the eigenvalues of the Laplacian of the adjacency matrix `A`.
"""
function laplacian_spectrum(A::AbstractMatrix{T}) where {T<:Real}
    return eigvals(laplacian(A))
end

"""
    algebraic_connectivity(A::AbstractMatrix[; normalized::Bool=false])

Computes the algebraic connectivity (second smallest Laplacian eigenvalue) of a graph defined by the adjacency matrix `A`. If `normalized` is true, it returns the normalized connectivity value divided by the maximum degree of the graph, otherwise it returns the raw algebraic connectivity value.
"""
function algebraic_connectivity(
    A::AbstractMatrix{T}; normalized::Bool=false
) where {T<:Real}
    λ = laplacian_spectrum(A)
    ac = λ[2]
    return normalized ? ac / maximum(degrees(A)) : ac
end
