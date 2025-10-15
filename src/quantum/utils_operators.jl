# creates a unitary matrix from the hamiltonian
function get_unitary(H::AbstractMatrix{T}, δt::Real) where {T<:Number}
    # h_eigvals, h_eigvecs = eigen(H)
    # return h_eigvecs * exp(Diagonal(-im * δt .* h_eigvals)) * h_eigvecs'
    return cis(-δt * H)
end

function exp_val(O::AbstractMatrix{T}, ψ::AbstractVector{S}) where {T<:Number,S<:Number}
    tmp = similar(ψ, promote_type(T, S))
    mul!(tmp, O, ψ)
    val = dot(ψ, tmp)
    return real(val)
end

function exp_val(O::AbstractMatrix{T}, ρ::AbstractMatrix{S}) where {T<:Number,S<:Number}
    val = sum(@views ρ .* transpose(O))
    return real(val)
end

function commute(A::AbstractMatrix{T}, B::AbstractMatrix{T}; kwargs...) where {T<:Number}
    return isapprox(A * B, B * A; kwargs...)
end

function is_unitary(U::AbstractMatrix{T}; tol::Real=DEFAULT_TOL) where {T<:Number}
    return opnorm(U' * U - I, Inf) <= tol && opnorm(U * U' - I, Inf) <= tol
end

function is_hermitian(H::AbstractMatrix{T}; tol::Real=DEFAULT_TOL) where {T<:Number}
    return ishermitian(H) || opnorm(H - H', Inf) <= tol
end

function is_psd(M::AbstractMatrix{T}; tol::Real=DEFAULT_TOL) where {T<:Number}
    H = Hermitian((M + M') / 2)
    return eigmin(H) >= -tol
end

function is_complete(
    povm::AbstractVector{<:AbstractMatrix{T}}; tol::Real=DEFAULT_TOL
) where {T<:Number}
    S = sum(povm)
    return opnorm(S - I, Inf) <= tol
end

function is_povm(
    povm::AbstractVector{<:AbstractMatrix{T}}; tol::Real=DEFAULT_TOL
) where {T<:Number}
    all(E -> is_psd(E; tol=tol), povm) || return false
    return is_complete(povm; tol=tol)
end
