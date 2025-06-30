"""
    train_epoch!(model::Flux.Chain, data::Flux.DataLoader, rule::NamedTuple, loss::Function)

Trains a Flux model for one epoch using a DataLoader for minibatching. Loss is computed using the provided loss function, rule is the optimization rule. This function uses [`Flux.jl`](https://fluxml.ai/Flux.jl/stable/).

Returns the average loss for the epoch.
"""
function train_epoch!(
    model::Flux.Chain, data::Flux.DataLoader, rule::NamedTuple, loss::Function
)
    epoch_loss = zero(Float32)
    for (x_batch, y_batch) in data
        loss_val, grads = Flux.withgradient(model) do m
            loss(m, x_batch, y_batch)
        end
        Flux.update!(rule, model, grads[1])
        epoch_loss += loss_val
    end
    return epoch_loss / data.batchsize
end

"""
    train_epoch!(model::Flux.Chain, data::Tuple{AbstractMatrix,Flux.OneHotArrays.OneHotMatrix}, rule::NamedTuple, loss::Function)

Method for a single batch of data provided as a tuple of input matrix and one-hot encoded labels.
"""
function train_epoch!(
    model::Flux.Chain,
    data::Tuple{AbstractMatrix,Flux.OneHotArrays.OneHotMatrix},
    rule::NamedTuple,
    loss::Function,
)
    loss_val, grads = Flux.withgradient(model) do m
        loss(m, data[1], data[2])
    end
    Flux.update!(rule, model, grads[1])
    return loss_val
end

"""
    nn_layer(data, labels_train, labels_test[, label_set, train_pos, test_pos; epochs, rate, batchsize, minibatch_shuffle, enable_bar])

Creates and trains a neural network layer on the provided data using the Adam optimizer, softmax activation, and cross-entropy loss function. This function uses [`Flux.jl`](https://fluxml.ai/Flux.jl/stable/) and a progressbar from [`ProgressMeter.jl`](https://github.com/timholy/ProgressMeter.jl).

# Arguments
* `data::AbstractMatrix`: Input data matrix where columns are samples
* `labels_train::AbstractVector`: Training labels
* `labels_test::AbstractVector`: Test labels
* `label_set=unique(labels_train)`: Set of unique labels (default: `unique(labels_train)`)
* `train_pos=1:60_000`: Indices for training data
* `test_pos=60_001:size(data, 2)`: Indices for test data
    ## Keyword Arguments
    * `epochs::Integer=100`: Number of training epochs
    * `learn_rate::AbstractFloat=0.01`: Learning rate for Adam optimizer
    * `batchsize::Integer=100`: Batch size for training (use 1 for full-batch)
    * `minibatch_shuffle::Bool=true`: Whether to shuffle mini-batches
    * `enable_bar::Bool=true`: Whether to show progress bar

Returns a nametuple containing:
* `model`: The trained neural network `Flux.Chain` model
* `losses`: Vector of loss values for each epoch
* `train_accuracies`: Vector of training accuracies for each epoch
* `test_accuracies`: Vector of test accuracies for each epoch
"""
function nn_layer(
    data::AbstractMatrix{T},
    labels_train::AbstractVector{J},
    labels_test::AbstractVector{J},
    label_set=unique(labels_train),
    train_pos=1:60_000,
    test_pos=60_001:size(data, 2);
    epochs::Integer=100,
    learn_rate::AbstractFloat=0.01,
    batchsize::Integer=100,
    minibatch_shuffle::Bool=true,
    enable_bar::Bool=true,
) where {T<:AbstractFloat,J<:Integer}
    data_train = data[:, train_pos]
    data_test = data[:, test_pos]

    data_size = size(data_train, 1)
    num_classes = length(label_set)
    nn_model = Chain(Dense(data_size, num_classes))
    loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

    opt_state = Flux.setup(Adam(learn_rate), nn_model)

    y_train_onehot = Flux.onehotbatch(labels_train, label_set)
    if batchsize == 1
        train_data_loader = (data_train, y_train_onehot)
    else
        train_data_loader = Flux.DataLoader(
            (data_train, y_train_onehot); batchsize=batchsize, shuffle=minibatch_shuffle
        )
    end

    epoch_loss = Vector{Float32}(undef, epochs)
    epoch_tr_acc = Vector{Float64}(undef, epochs)
    epoch_te_acc = Vector{Float64}(undef, epochs)
    metrics = Dict(
        "Loss" => epoch_loss,
        "Train Accuracy" => epoch_tr_acc,
        "Test Accuracy" => epoch_te_acc,
    )

    epoch_progress = Progress(epochs; desc="Training: ", enabled=enable_bar)
    for epoch in 1:epochs
        epoch_loss[epoch] = train_epoch!(nn_model, train_data_loader, opt_state, loss_fn)
        epoch_tr_acc[epoch] = accuracy(nn_model, data_train, label_set, labels_train)
        epoch_te_acc[epoch] = accuracy(nn_model, data_test, label_set, labels_test)
        next!(epoch_progress; showvalues=ml_showvalues(epoch, metrics))
    end

    return (
        model=nn_model,
        losses=epoch_loss,
        train_accuracies=epoch_tr_acc,
        test_accuracies=epoch_te_acc,
    )
end
