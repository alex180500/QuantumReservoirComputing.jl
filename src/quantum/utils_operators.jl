# creates a unitary matrix from the hamiltonian
function get_unitary(H::AbstractMatrix{T}, δt::Real) where {T<:Number}
    # h_eigvals, h_eigvecs = eigen(H)
    # return h_eigvecs * exp(Diagonal(-im * δt .* h_eigvals)) * h_eigvecs'
    return cis(-δt * H)
end

function commute(A::AbstractMatrix{T}, B::AbstractMatrix{T}; kwargs...) where {T<:Number}
    return isapprox(A * B, B * A; kwargs...)
end

function isunitary(U::AbstractMatrix{T}; tol::Real=DEFAULT_TOL) where {T<:Number}
    return isapprox(U * U', I; atol=tol)
end

function exp_val(O::AbstractMatrix{T}, ψ::AbstractVector{S}) where {T<:Number,S<:Number}
    tmp = similar(ψ, promote_type(T, S))
    mul!(tmp, O, ψ)
    return real(dot(ψ, tmp))
end

function exp_val(O::AbstractMatrix{T}, ρ::AbstractMatrix{S}) where {T<:Number,S<:Number}
    return real(tr(ρ * O))
end
