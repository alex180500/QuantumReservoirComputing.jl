# given a matrix or vector, this function returns the number of qubits
function get_nsys(mat::AbstractVecOrMat{T}) where {T<:Number}
    return Int(log2(size(mat, 1)))
end

# given a matrix or vector, this function returns the number of qudits of dimension d
function get_nsys(mat::AbstractVecOrMat{T}, d::Integer) where {T<:Number}
    return Int(log(d, size(mat, 1)))
end

# creates an identity matrix of dimension dim
@memoize eye(dim::Integer) = Matrix{ComplexF64}(I, dim, dim)
@memoize eye_qubits(n_sys::Integer) = eye(2^n_sys)
# creates a maximally mixed state of dimension dim
@memoize max_mixed(dim::Integer) = eye(dim) / dim

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
