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
