# TODO this macro works but it slows down the training loop 
macro trainprogress(metrics_expr, args...)
    if length(args) == 1
        enabled_expr = true
    else
        enabled_expr = args[1]
    end
    loop_expr = args[end]
    loop_iter = loop_expr.args[1]
    loop_body = loop_expr.args[2]
    iter_var = loop_iter.args[1]
    iter_range = loop_iter.args[2]
    return quote
        local metrics_dict = $(esc(metrics_expr))
        local epoch_progress = Progress(
            length($(esc(iter_range))),
            desc="Training: ",
            enabled=$(esc(enabled_expr))
        )
        for $(esc(iter_var)) in $(esc(iter_range))
            $(esc(loop_body))
            next!(
                epoch_progress,
                showvalues=ml_showvalues($(esc(iter_var)), metrics_dict)
            )
        end
    end
end

function ml_showvalues(
    epoch::Integer,
    metrics::Dict{String,<:AbstractVector};
    n_digits::Integer=4
)
    vec_metrics = Vector{Tuple{String,Any}}([("Epoch", epoch)])
    for (key, value_arr) in metrics
        value = round(value_arr[epoch], digits=n_digits)
        push!(vec_metrics, (key, value))
    end
    () -> vec_metrics
end
