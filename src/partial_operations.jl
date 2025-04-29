# partial trace for 2 qubits system
function ptrace(ρ::AbstractMatrix{T}, keep::Int) where {T<:Number}
    ρ_tensor = reshape(ρ, 2, 2, 2, 2)
    red_ρ = Matrix{T}(undef, 2, 2)
    if keep == 1
        @inbounds for i in 1:2, j in 1:2
            red_ρ[i, j] = ρ_tensor[1, i, 1, j] + ρ_tensor[2, i, 2, j]
        end
    elseif keep == 2
        @inbounds for i in 1:2, j in 1:2
            red_ρ[i, j] = ρ_tensor[i, 1, j, 1] + ρ_tensor[i, 2, j, 2]
        end
    else
        throw(ArgumentError("keep must be 1 or 2"))
    end
    return red_ρ
end

# partial trace for N qudits, where only one system is kept
function ptrace(
    ρ::AbstractMatrix{T},
    keep::Int,
    n_sys::Int;
    d::Int=2
) where {T<:Number}
    return ptrace(ρ, (keep,), n_sys, d=d)
end

# partial trace for N qudits system with n_sys calculated automatically
# from the size of the density matrix
function ptrace(
    ρ::AbstractMatrix{T},
    ikeep::NTuple{N,Int};
    d::Int=2
) where {N,T<:Number}
    n_sys = Int(log(d, size(ρ, 1)))
    return ptrace(ρ, ikeep, n_sys, d=d)
end

# partial trace for N qudits system
function ptrace(
    ρ::AbstractMatrix{T},
    ikeep::NTuple{N,Int},
    n_sys::Int;
    d::Int=2,
) where {N,T<:Number}
    keep = ntuple(k -> n_sys - ikeep[k] + 1, N)
    locs = Tuple(x for x in n_sys:-1:1 if !(x in keep))
    strides = ntuple(i -> d^(i - 1), n_sys)
    out_strides = ntuple(i -> d^(i - 1), N)
    remain_strides = ntuple(i -> strides[keep[N-i+1]], N)
    trace_strides = ntuple(i -> strides[locs[i]], n_sys - N)
    state = zeros(ComplexF64, d^N, d^N)
    _ptrace_dim!(
        Val{d}(),
        state,
        ρ,
        trace_strides,
        out_strides,
        remain_strides
    )
    return state
end

# most generic partial trace, works for any num of systems and dimensions
function ptrace(
    ρ::AbstractMatrix{T},
    ikeep::NTuple{N,Int},
    idims::NTuple{M,Int}
) where {N,M,T<:Number}
    @inbounds keepdim = prod(idims[k] for k in ikeep)
    keep::NTuple{N,Int} = ntuple(k -> M - ikeep[k] + 1, N)

    traceidx::NTuple{2 * M,Int} =
        ntuple(k -> (k > M ? k - M : k) + (k in keep ? M : 0), 2 * M)

    dims::NTuple{M,Int} = reverse(idims)
    tensor::Array{T,2 * M} = reshape(ρ, dims..., dims...)

    traced = tensortrace(tensor, traceidx)
    return reshape(traced, keepdim, keepdim)
end

# private partial trace optimized function taken from https://github.com/QuantumBFS/Yao.jl
@generated function _ptrace_dim!(
    ::Val{D},
    out::AbstractMatrix,
    dm::AbstractMatrix,
    trace_strides::NTuple{K,Int},
    out_strides::NTuple{M,Int},
    remain_strides::NTuple{M,Int}
) where {D,K,M}
    quote
        sumc = length(remain_strides) == 0 ? 1 : 1 - sum(remain_strides)
        suma = length(out_strides) == 0 ? 1 : 1 - sum(out_strides)
        Base.Cartesian.@nloops(
            $M,
            i,
            d -> 1:$D,
            d -> (@inbounds sumc += i_d * remain_strides[d];
            @inbounds suma += i_d * out_strides[d]), # PRE
            d -> (@inbounds sumc -= i_d * remain_strides[d];
            @inbounds suma -= i_d * out_strides[d]), # POST
            begin # BODY
                sumd =
                    length(remain_strides) == 0 ? 1 : 1 - sum(remain_strides)
                sumb = length(out_strides) == 0 ? 1 : 1 - sum(out_strides)
                Base.Cartesian.@nloops(
                    $M,
                    j,
                    d -> 1:$D,
                    d -> (@inbounds sumd += j_d * remain_strides[d];
                    @inbounds sumb += j_d * out_strides[d]), # PRE
                    d -> (@inbounds sumd -= j_d * remain_strides[d];
                    @inbounds sumb -= j_d * out_strides[d]), # POST
                    begin
                        sume =
                            length(trace_strides) == 0 ? 1 :
                            1 - sum(trace_strides)
                        Base.Cartesian.@nloops(
                            $K,
                            k,
                            d -> 1:$D,
                            d -> (@inbounds sume += k_d * trace_strides[d]), # PRE
                            d -> (@inbounds sume -= k_d * trace_strides[d]), # POST
                            @inbounds out[suma, sumb] +=
                                dm[sumc+sume-1, sumd+sume-1]
                        )
                    end
                )
            end
        )
    end
end
