# most generic partial trace, works for any num of systems and dimensions
# taken from https://github.com/iitis/QuantumInformation.jl

function ptrace(
    ρ::AbstractMatrix{T},
    ikeep::NTuple{K,Int},
    idims::NTuple{M,Int}
) where {K,M,T<:Number}
    @inbounds keepdim = prod(idims[k] for k in ikeep)
    keep::NTuple{K,Int} = ntuple(k -> M - ikeep[k] + 1, K)

    traceidx::NTuple{2 * M,Int} =
        ntuple(k -> (k > M ? k - M : k) + (k in keep ? M : 0), 2 * M)

    dims::NTuple{M,Int} = reverse(idims)
    tensor::Array{T,2 * M} = reshape(ρ, dims..., dims...)

    traced = tensortrace(tensor, traceidx)
    return reshape(traced, keepdim, keepdim)
end

# partial trace for N qudits system

function ptrace(
    ρ::AbstractMatrix{T},
    keep::Int,
    n_sys::Int=get_nsys(ρ, d);
    d::Int
) where {T<:Number}
    return ptrace(ρ, (keep,), n_sys, d=d)
end

function ptrace(
    ρ::AbstractMatrix{T},
    ikeep::NTuple{K,Int},
    n_sys::Int=get_nsys(ρ, d);
    d::Int,
) where {K,T<:Number}
    keep = ntuple(k -> n_sys - ikeep[k] + 1, K)
    strides = ntuple(i -> d^(i - 1), n_sys)
    out_strides = ntuple(i -> d^(i - 1), K)
    remain_strides = ntuple(i -> strides[keep[K-i+1]], K)
    trace_strides = ntuple(j -> begin
        cnt = 0
        for x in n_sys:-1:1
            if x ∉ keep
                cnt += 1
                if cnt == j
                    return strides[x]
                end
            end
        end
    end, n_sys - K)
    state = zeros(ComplexF64, d^K, d^K)
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

# specialized partial trace for qubits systems

function ptrace_qubits(
    ρ::AbstractMatrix{T},
    keep::Int,
    n_sys::Int=get_nsys(ρ)
) where {T<:Number}
    return ptrace_qubits(ρ, (keep,), n_sys)
end

function ptrace_qubits(
    ρ::AbstractMatrix{T},
    ikeep::NTuple{K,Int},
    n_sys::Int=get_nsys(ρ)
) where {K,T<:Number}
    keep = ntuple(k -> n_sys - ikeep[k] + 1, K)
    strides = ntuple(i -> 1 << (i - 1), n_sys)
    out_strides = ntuple(i -> 1 << (i - 1), K)
    remain_strides = ntuple(i -> strides[keep[K-i+1]], K)
    trace_strides = ntuple(j -> begin
        cnt = 0
        for x in n_sys:-1:1
            if x ∉ keep
                cnt += 1
                if cnt == j
                    return strides[x]
                end
            end
        end
    end, n_sys - K)
    state = zeros(ComplexF64, 1 << K, 1 << K)
    _ptrace_dim!(
        Val{2}(),
        state,
        ρ,
        trace_strides,
        out_strides,
        remain_strides
    )
    return state
end

# partial trace that only gets the diagonal

function ptrace_d(
    ρ::AbstractMatrix{T},
    keep::Int,
    n_sys::Int=get_nsys(ρ, d);
    d::Int
) where {T<:Number}
    return ptrace_d(ρ, (keep,), n_sys, d=d)
end

function ptrace_d(
    ρ::AbstractMatrix{T},
    ikeep::NTuple{K,Int},
    n_sys::Int=get_nsys(ρ, d);
    d::Int
) where {K,T<:Number}
    keep = ntuple(k -> n_sys - ikeep[k] + 1, K)
    strides = ntuple(i -> d^(i - 1), n_sys)
    out_strides = ntuple(i -> d^(i - 1), K)
    remain_strides = ntuple(i -> strides[keep[K-i+1]], K)
    trace_strides = ntuple(j -> begin
        cnt = 0
        for x in n_sys:-1:1
            if x ∉ keep
                cnt += 1
                if cnt == j
                    return strides[x]
                end
            end
        end
    end, n_sys - K)
    diag = zeros(Float64, d^K)
    _ptrace_diag_dim!(
        Val{d}(),
        diag,
        ρ,
        trace_strides,
        out_strides,
        remain_strides
    )
    return diag
end

# specialized partial trace for qubits systems that only gets the diagonal

function ptrace_qubits_d(
    ρ::AbstractMatrix{T},
    keep::Int,
    n_sys::Int=get_nsys(ρ)
) where {T<:Number}
    return ptrace_qubits_d(ρ, (keep,), n_sys)
end

function ptrace_qubits_d(
    ρ::AbstractMatrix{T},
    ikeep::NTuple{K,Int},
    n_sys::Int=get_nsys(ρ)
) where {K,T<:Number}
    keep = ntuple(k -> n_sys - ikeep[k] + 1, K)
    strides = ntuple(i -> 1 << (i - 1), n_sys)
    out_strides = ntuple(i -> 1 << (i - 1), K)
    remain_strides = ntuple(i -> strides[keep[K-i+1]], K)
    trace_strides = ntuple(j -> begin
        cnt = 0
        for x in n_sys:-1:1
            if x ∉ keep
                cnt += 1
                if cnt == j
                    return strides[x]
                end
            end
        end
    end, n_sys - K)
    state = zeros(Float64, 1 << K)
    _ptrace_diag_dim!(
        Val{2}(),
        state,
        ρ,
        trace_strides,
        out_strides,
        remain_strides
    )
    return state
end

# partial trace for 2 qubits system

function ptrace_2qubits(ρ::AbstractMatrix{T}, keep::Int) where {T<:Number}
    ρ_tensor = reshape(ρ, 2, 2, 2, 2)
    red_ρ = Matrix{T}(undef, 2, 2)
    if keep == 1
        @inbounds for i in 1:2, j in 1:2
            red_ρ[i, j] = ρ_tensor[1, i, 1, j] + ρ_tensor[2, i, 2, j]
        end
    else
        @inbounds for i in 1:2, j in 1:2
            red_ρ[i, j] = ρ_tensor[i, 1, j, 1] + ρ_tensor[i, 2, j, 2]
        end
    end
    return red_ρ
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

@generated function _ptrace_diag_dim!(
    ::Val{D},
    diag::AbstractVector,
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
                sume =
                    length(trace_strides) == 0 ? 1 : 1 - sum(trace_strides)
                Base.Cartesian.@nloops(
                    $K,
                    k,
                    d -> 1:$D,
                    d -> (@inbounds sume += k_d * trace_strides[d]), # PRE
                    d -> (@inbounds sume -= k_d * trace_strides[d]), # POST
                    @inbounds diag[suma] += dm[sumc+sume-1, sumc+sume-1]
                )
            end
        )
    end
end
