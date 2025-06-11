# given an integer number in base 10, this function converts it to a binary string
# and returns the element in position idx
function get_bit(num::Integer, idx::Integer)
    return (num >> (idx - 1)) & 1
end

# create a BitMatrix where a column is the binary representation of the integer i
# for i = 0, 1, ..., 2^N-1
function get_bit_table(num::Integer)
    M = 2^num
    # create the vector 0,1,…,2^N–1 as UInt
    indices = UInt.(0:M-1)
    # build masks for each bit position (MSB first)
    masks = UInt.(1) .<< ((num-1):-1:0)
    # allocate the BitMatrix
    bit_table = BitMatrix(undef, num, M)
    # fill each row j by testing the j-th mask against every index
    @inbounds for j in 1:num
        # (indices .& masks[j]) .!= 0 is a Bool-vector of length M
        bit_table[j, :] .= (indices .& masks[j]) .!= 0
    end
    return bit_table
end
