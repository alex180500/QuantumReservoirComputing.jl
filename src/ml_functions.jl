# loads some data and performs PCA on it
function pca_analysis(
    train_data::AbstractMatrix{<:Real},
    test_data::AbstractMatrix{<:Real},
    k::Union{Int,Nothing}=nothing;
    pratio::Float64=0.99
)
    if isnothing(k)
        pca_model = fit(PCA, train_data, pratio=pratio)
    else
        pca_model = fit(PCA, train_data, maxoutdim=k, pratio=pratio)
    end
    train_transformed = transform(pca_model, train_data)
    test_transformed = transform(pca_model, test_data)
    return train_transformed, test_transformed
end
