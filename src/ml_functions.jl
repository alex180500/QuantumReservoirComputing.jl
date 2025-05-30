# loads some data and performs PCA on it
function pca_analysis(train_data, test_data)
    pca_model = fit(PCA, train_data)
    train_transformed = transform(pca_model, train_data)
    test_transformed = transform(pca_model, test_data)    
    return train_transformed, test_transformed
end
