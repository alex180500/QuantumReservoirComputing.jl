# this gets the size of a particular object in MB
function get_mb(item)
    return Base.summarysize(item) / 1e6
end

# given an integer number in base 10, this function converts it to a binary string
# and returns the element in position idx
function get_bit(num::Integer, idx::Integer)
    return (num >> (idx - 1)) & 1
end

# given a matrix, this function returns the number of qubits
function get_nsys(mat::AbstractMatrix{T}) where {T<:Number}
    return Int(log2(size(mat, 1)))
end

# given a matrix, this function returns the number of qudits of dimension d
function get_nsys(mat::AbstractMatrix{T}, d::Int) where {T<:Number}
    return Int(log(d, size(mat, 1)))
end