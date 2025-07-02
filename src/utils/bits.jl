# given an integer number in base 10, this function converts it to a binary string
# and returns the element in position idx
function get_bit(num::Integer, idx::Integer)
    return (num >> (idx - 1)) & 1
end

# create a BitMatrix where a column is the binary representation of the integer i
# for i = 0, 1, ..., 2^num-1
function get_bit_table(num::Integer)
    M = 2^num
    indices = UInt.(0:(M - 1))
    masks = UInt.(1) .<< ((num - 1):-1:0)
    bit_table = BitMatrix(undef, num, M)
    @inbounds for j in 1:num
        bit_table[j, :] .= (indices .& masks[j]) .!= 0
    end
    return bit_table
end
