# Internal default tolerance (unexported)
const DEFAULT_TOL::Float64 = 1e-8

# this gets the size of a particular object in MB
function get_MB(item)
    return Base.summarysize(item) / 1e6
end

function get_mean_last(
    data::AbstractVector{T}, last_n::Integer; return_std::Bool=false
) where {T<:Real}
    chosen_data = data[(end - last_n + 1):end]
    if return_std
        return mean(chosen_data), std(chosen_data)
    end
    return mean(chosen_data)
end

function get_mean_last(
    data::AbstractMatrix{T}, last_n::Integer; return_std::Bool=false
) where {T<:Real}
    if return_std
        data_vec = get_mean_last.(eachcol(data), last_n)
        return mean(data_vec), std(data_vec)
    end
    return mean(data[(end - last_n + 1):end, :])
end

# counts the number of unique integers in a vector and indices
function count_unique_all(arr::AbstractVector{J}) where {J<:Integer}
    m, M = extrema(arr)
    cu = zeros(Int, M - m + 1)
    @inbounds for i in eachindex(arr)
        cu[arr[i] - m + 1] += 1
    end
    return cu
end

count_unique(arr::AbstractArray{T}) where {T<:Real} = countmap(arr)

function unique_indices(arr::AbstractArray{T}) where {T<:Real}
    unique_vals = unique(arr)
    return Dict{T,Vector{Int}}(val => findall(==(val), arr) for val in unique_vals)
end

# some functions for approximate values
function unique_approx(arr::AbstractArray{T}; tol::Real=DEFAULT_TOL) where {T<:Real}
    sorted_arr = sort(arr)
    unique_vals = [sorted_arr[1]]
    for x in sorted_arr[2:end]
        if abs(x - last(unique_vals)) > tol
            push!(unique_vals, x)
        end
    end
    return unique_vals
end

function count_unique_approx(arr::AbstractArray{T}; tol::Real=DEFAULT_TOL) where {T<:Real}
    unique_vals = unique_approx(arr; tol=tol)
    return Dict{T,Int}(val => count(x -> abs(x - val) < tol, arr) for val in unique_vals)
end

function unique_indices_approx(arr::AbstractArray{T}; tol::Real=DEFAULT_TOL) where {T<:Real}
    unique_vals = unique_approx(arr; tol=tol)
    return Dict{T,Vector{Int}}(
        val => findall(x -> abs(x - val) < tol, arr) for val in unique_vals
    )
end
