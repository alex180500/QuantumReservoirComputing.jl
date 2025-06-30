"""
    @trainprogress(metrics_expr, args...)

Macro that adds a progress bar to training loops with metric display using [`ml_showvalues`](@ref). It takes a metrics dictionary of values that are updated inside the loop (and that will be showed) and then a for loop, showing progress with current metric values. Uses [`ProgressMeter.jl`](https://github.com/timholy/ProgressMeter.jl).

!!! todo
    This macro is currently slower than the current approach in [`nn_layer`](@ref) and might be optimized. Still, it can be useful for debugging.
"""
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
            length($(esc(iter_range))); desc="Training: ", enabled=($(esc(enabled_expr)))
        )
        for $(esc(iter_var)) in $(esc(iter_range))
            $(esc(loop_body))
            next!(epoch_progress; showvalues=ml_showvalues($(esc(iter_var)), metrics_dict))
        end
    end
end

"""
    ml_showvalues(epoch::Integer, metrics::Dict{String,<:AbstractVector}[; n_digits::Integer=4])

Formats training metrics for display for the progress bar. It truncates the values to a specified number of digits for better readability.

Returns a function that provides metric values rounded to specified digits.
"""
function ml_showvalues(
    epoch::Integer, metrics::Dict{String,<:AbstractVector}; n_digits::Integer=4
)
    vec_metrics = Vector{Tuple{String,Any}}([("Epoch", epoch)])
    for (key, value_arr) in metrics
        value = round(value_arr[epoch]; digits=n_digits)
        push!(vec_metrics, (key, value))
    end
    () -> vec_metrics
end
