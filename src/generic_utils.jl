# utils functions

function get_mb(item)
    return Base.summarysize(item) / 1e6
end

function get_bit(num::Integer, idx::Integer)
    return (num >> (idx - 1)) & 1
end
