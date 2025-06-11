# loads some data and performs PCA on it
function pca_analysis(
    train_data::AbstractMatrix{T},
    test_data::AbstractMatrix{T},
    k::Union{Int,Nothing}=nothing;
    pratio::Real=0.99,
    return_model::Bool=false
) where {T<:Real}
    if isnothing(k)
        pca_model = fit(PCA, train_data, pratio=pratio)
    else
        pca_model = fit(PCA, train_data, maxoutdim=k, pratio=pratio)
    end
    train_transformed = transform(pca_model, train_data)
    test_transformed = transform(pca_model, test_data)
    if return_model
        return pca_model, train_transformed, test_transformed
    else
        return train_transformed, test_transformed
    end
end

function rescale_data(
    train_data::AbstractMatrix{T},
    test_data::AbstractMatrix{T};
    clamping::Bool=true
) where {T<:Real}
    val_min, val_max = extrema(train_data)
    scaled_train = (train_data .- val_min) ./ (val_max - val_min)
    scaled_test = (test_data .- val_min) ./ (val_max - val_min)
    if clamping
        scaled_test = clamp.(scaled_test, 0, 1)
    end
    return scaled_train, scaled_test
end