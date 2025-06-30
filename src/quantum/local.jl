const LocalOperators{N,T<:Number} = NTuple{N,<:AbstractMatrix{T}}

function LocalOperators{N,T}(::UndefInitializer) where {N,T<:Number}
    dim = 2^N
    return ntuple(i -> Matrix{T}(undef, dim, dim), N)
end

function LocalOperators{N,T}(::typeof(zeros)) where {N,T<:Number}
    dim = 2^N
    return ntuple(i -> zeros(T, dim, dim), N)
end

# given an operator op construct a list of local operators
# for each qubit in the system
function LocalOperators{N,T}(op::AbstractMatrix) where {N,T<:Number}
    all_ops = LocalOperators{N,T}(undef)
    kron!(all_ops[1], op, eye_qubits(N - 1))
    @inbounds for idx in 2:(N - 1)
        right_part = kron(op, eye_qubits(N - idx))
        kron!(all_ops[idx], eye_qubits(idx - 1), right_part)
    end
    kron!(all_ops[N], eye_qubits(N - 1), op)
    return all_ops
end

# Convenience constructor that infers type
function LocalOperators(op::AbstractMatrix{T}, n_sys::Integer) where {T<:Number}
    return LocalOperators{n_sys,T}(op)
end

function LocalOperators{N}(op::AbstractMatrix{T}) where {N,T<:Number}
    return LocalOperators{N,T}(op)
end
