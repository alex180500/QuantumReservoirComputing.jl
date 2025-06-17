const LocalOperators{T<:Number} = AbstractVector{<:AbstractMatrix{T}}

function LocalOperators{T}(::UndefInitializer, n::Integer) where {T<:Number}
    dim = 2^n
    return [Matrix{T}(undef, dim, dim) for _ in 1:n]
end

# given an operator op construct a list of local operators
# for each qubit in the system
function LocalOperators{T}(
    op::AbstractMatrix,
    n_sys::Integer
) where {T<:Number}
    all_ops = LocalOperators{T}(undef, n_sys)
    kron!(all_ops[1], op, eye_qubits(n_sys - 1))
    @inbounds for idx in 2:n_sys-1
        right_part = kron(op, eye_qubits(n_sys - idx))
        kron!(all_ops[idx], eye_qubits(idx - 1), right_part)
    end
    kron!(all_ops[n_sys], eye_qubits(n_sys - 1), op)
    return all_ops
end

# Convenience constructor that infers type
LocalOperators(op::AbstractMatrix{T}, n_sys::Integer) where {T<:Number} =
    LocalOperators{T}(op, n_sys)
