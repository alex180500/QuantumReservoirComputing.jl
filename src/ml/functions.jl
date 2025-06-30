"""
    pca_analysis(train_data::AbstractMatrix, test_data::AbstractMatrix[, k::Union{Int,Nothing}=nothing; pratio=0.99, return_model=false])

Performs Principal Component Analysis on training data and transforms both training and test data. `k` is the number of PCA components to keep, if `k` is `nothing` it will use the `pratio` parameter to determine the number of components that explain at least `pratio` of the variance. This function uses [`MultivariateStats.jl`](https://juliastats.org/MultivariateStats.jl/stable/pca/).

Returns `(train_transformed, test_transformed)` or, if `return_model` is true, `(pca_model, train_transformed, test_transformed)`.
"""
function pca_analysis(
    train_data::AbstractMatrix{T},
    test_data::AbstractMatrix{T},
    k::Union{Int,Nothing}=nothing;
    pratio::Real=0.99,
    return_model::Bool=false,
) where {T<:Real}
    if isnothing(k)
        pca_model = fit(PCA, train_data; pratio=pratio)
    else
        pca_model = fit(PCA, train_data; maxoutdim=k, pratio=pratio)
    end
    train_transformed = transform(pca_model, train_data)
    test_transformed = transform(pca_model, test_data)
    if return_model
        return pca_model, train_transformed, test_transformed
    else
        return train_transformed, test_transformed
    end
end

"""
    rescale_data(train_data::AbstractMatrix, test_data::AbstractMatrix[; clamping=true])

Rescales training and test data to ``[0,1]`` range based on training data extremas. Optionally clamps test data to ``[0,1]`` range if it falls outside due to different distributions.

Returns `(scaled_train, scaled_test)`.
"""
function rescale_data(
    train_data::AbstractMatrix{T}, test_data::AbstractMatrix{T}; clamping::Bool=true
) where {T<:Real}
    val_min, val_max = extrema(train_data)
    scaled_train = (train_data .- val_min) ./ (val_max - val_min)
    scaled_test = (test_data .- val_min) ./ (val_max - val_min)
    if clamping
        scaled_test = clamp.(scaled_test, 0, 1)
    end
    return scaled_train, scaled_test
end


"""
    accuracy(model::Flux.Chain, x::AbstractMatrix, classes::AbstractVector, y::AbstractVector)

Calculates the accuracy of a Flux model on given data `x` comparing with labels `y`. This function uses [`Flux.jl`](https://fluxml.ai/Flux.jl/stable/).

Returns the fraction of correctly predicted labels.
"""
function accuracy(
    model::Flux.Chain,
    x::AbstractMatrix{T},
    classes::AbstractVector{I},
    y::AbstractVector{I},
) where {T<:AbstractFloat,I<:Integer}
    return mean(Flux.onecold(model(x), classes) .== y)
end
