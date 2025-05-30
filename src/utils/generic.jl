# this gets the size of a particular object in MB
function get_mb(item)
    return Base.summarysize(item) / 1e6
end

# counts the number of unique integers in a vector
function count_unique(arr::Vector{Int}, M::Int=maximum(arr))
    cu = zeros(Int, M)
    @inbounds for i in eachindex(arr)
        cu[arr[i]] += 1
    end
    return cu
end
