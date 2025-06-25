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

# calculates a random Haar distributed unitary matrix
function haar_unitary(n::Integer=2)
    Z = randn(ComplexF64, n, n)
    QR_decomp = qr!(Z)
    Λ = diag(QR_decomp.R)
    @. Λ /= abs(Λ)
    return QR_decomp.Q * Diagonal(Λ)
end

# creates a random U matrix and then a random state
# as  U |0⟩
function haar_state(n::Integer=2)
    Z = randn(ComplexF64, n, n)
    QR_decomp = qr!(Z)
    return QR_decomp.Q[:, 1]
end

# creates a random pure state and then the density matrix
function haar_dm(n::Integer=2)
    ψ::Vector{ComplexF64} = haar_state(n)
    return ψ * ψ'
end
