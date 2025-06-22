# this gets the size of a particular object in MB
function get_mb(item)
    return Base.summarysize(item) / 1e6
end

# counts the number of unique integers in a vector
function count_unique(
    arr::AbstractVector{T},
    M::T=maximum(arr)
) where {T<:Integer}
    cu = zeros(Int, M)
    @inbounds for i in eachindex(arr)
        cu[arr[i]] += 1
    end
    return cu
end

function eigvals_2(mat::AbstractMatrix{T}) where {T<:Number}
    @inbounds begin
        a = mat[1, 1]
        b = mat[1, 2]
        d = mat[2, 2]
    end
    δ = a - d
    disc = sqrt(muladd(δ, δ, 4 * abs2(b)))
    t = (a + d)
    λ1 = (t - disc) / 2
    λ2 = (t + disc) / 2
    return Float64[λ1, λ2]
end

function get_mean_last(
    data::AbstractVector{T},
    last_n::Integer=10
) where {T<:Real}
    mean(data[end-last_n:end])
end

function get_average_data(
    data::AbstractMatrix{T},
    last_n::Integer=10;
    return_std::Bool=true
) where {T<:Real}
    n_ensemble = size(data, 2)
    data_vec = Vector{Float64}(undef, n_ensemble)
    @inbounds for i in eachindex(data_vec)
        data_vec[i] = get_mean_last(data[:, i], last_n)
    end
    if return_std
        return mean(data_vec), std(data_vec)
    end
    return mean(data_vec), std(data_vec)
end
