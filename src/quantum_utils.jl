# creates a maximally mixed state of dimension dim
function max_mixed(dim::Integer)
    return Matrix{ComplexF64}(I, dim, dim) / dim
end

# creates an identity matrix of dimension dim
function eye(dim::Integer)
    return Matrix{ComplexF64}(I, dim, dim)
end

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
