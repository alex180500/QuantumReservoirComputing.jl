# this gets the size of a particular object in MB
function get_mb(item)
    return Base.summarysize(item) / 1e6
end

function get_mean_last(data::AbstractVector{T}, last_n::Integer) where {T<:Real}
    return mean(data[(end - last_n):end])
end

function get_average_data(
    data::AbstractMatrix{T}, last_n::Integer=size(data, 1); return_std::Bool=true
) where {T<:Real}
    n_ensemble = size(data, 2)
    data_vec = Vector{Float64}(undef, n_ensemble)
    @inbounds for i in eachindex(data_vec)
        data_vec[i] = get_mean_last(data[:, i], last_n)
    end
    if return_std
        return mean(data_vec), std(data_vec)
    end
    return mean(data_vec)
end

# counts the number of unique integers in a vector
function count_unique(arr::AbstractVector{T}) where {T<:Integer}
    m, M = extrema(arr)
    cu = zeros(Int, M - m + 1)
    @inbounds for i in eachindex(arr)
        cu[arr[i] - m + 1] += 1
    end
    return cu
end

function count_unique(arr::AbstractVector{T}, format::Symbol) where {T<:Integer}
    cu = count_unique(arr)
    start_idx = minimum(arr) - 1
    if format == :pairs
        return Pair.(eachindex(cu) .+ start_idx, cu)
    elseif format == :dict
        return Dict{T,Int}(i + start_idx => cu[i] for i in eachindex(cu))
    else
        return cu
    end
end

count_unique(arr::AbstractVector) = countmap(arr)

function unique_indices(arr::AbstractArray{T}) where {T<:Real}
    unique_vals = unique(arr)
    return Dict{T,Vector{Int}}(val => findall(==(val), arr) for val in unique_vals)
end

function unique_approx(arr::AbstractArray{T}, tol::Real=1e-8) where {T<:Real}
    sorted_arr = sort(arr)
    unique_vals = [sorted_arr[1]]
    for x in sorted_arr[2:end]
        if abs(x - last(unique_vals)) > tol
            push!(unique_vals, x)
        end
    end
    return unique_vals
end

function count_unique_approx(arr::AbstractArray{T}, tol::Real=1e-8) where {T<:Real}
    unique_vals = unique_approx(arr, tol)
    return Dict{T,Int}(val => count(x -> abs(x - val) < tol, arr) for val in unique_vals)
end

function unique_indices_approx(arr::AbstractArray{T}, tol::Real=1e-8) where {T<:Real}
    unique_vals = unique_approx(arr, tol)
    return Dict{T,Vector{Int}}(
        val => findall(x -> abs(x - val) < tol, arr) for val in unique_vals
    )
end
