# calculates a random Haar distributed unitary matrix
function haar_unitary(n::Integer=2)
    Z = randn(ComplexF64, n, n)
    QR_decomp = qr!(Z)
    Λ = diag(QR_decomp.R)
    @. Λ /= abs(Λ)
    return QR_decomp.Q * Diagonal(Λ)
end

# creates a random U matrix and then a random state as  U |0⟩
function haar_state(n::Integer=2)
    Z = randn(ComplexF64, n, n)
    QR_decomp = qr!(Z)
    return QR_decomp.Q[:, 1]
end

# creates a random pure state and then the density matrix
function haar_dm(n::Integer=2)
    ψ = haar_state(n)::Vector{ComplexF64}
    return ψ * ψ'
end

function rand_symmetric_unitary(
    all_S::Union{AbstractVector{<:AbstractMatrix{T}},Tuple{Vararg{<:AbstractMatrix{T}}}};
    tol::Real=DEFAULT_TOL,
) where {T<:Number}
    combination_ops = sum(rand() * op for op in all_S)
    return rand_symmetric_unitary(combination_ops; tol=tol)
end

function rand_symmetric_unitary(
    S::AbstractMatrix{T}, n::Integer; tol::Real=DEFAULT_TOL
) where {T<:Number}
    blocks, V = get_symmetry_blocks(S; tol)
    val_blocks = values(blocks)
    return [rand_symmetric_unitary(val_blocks, V) for _ in 1:n]
end

function rand_symmetric_unitary(
    S::AbstractMatrix{T}; tol::Real=DEFAULT_TOL
) where {T<:Number}
    blocks, V = get_symmetry_blocks(S; tol)
    return rand_symmetric_unitary(values(blocks), V)
end

function get_symmetry_blocks(S::AbstractMatrix{T}; tol::Real=DEFAULT_TOL) where {T<:Number}
    eigen_decomp = eigen(S)
    eig_vals = real(eigen_decomp.values)
    blocks = unique_indices_approx(eig_vals; tol=tol)
    return blocks, eigen_decomp.vectors
end

function rand_symmetric_unitary(
    block_indices::Union{AbstractVector{<:AbstractVector{J}},Base.ValueIterator},
    V::AbstractMatrix{T},
) where {J<:Integer,T<:Number}
    Ublock = zero(V)
    for idxs in block_indices
        U_sub = haar_unitary(length(idxs))
        Ublock[idxs, idxs] = U_sub
    end
    return V * Ublock * V'
end
