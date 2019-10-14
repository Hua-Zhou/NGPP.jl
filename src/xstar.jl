struct NNGPXstar{T<:AbstractFloat} <: AbstractMatrix{T}
    nngp::NNGP{T}
    tmpn::Vector{T}
    tmpp::Vector{T}
    tmpr::Vector{T}
end

function NNGPXstar(nngp::NNGP{T}) where T<:AbstractFloat
    tmpn = Vector{T}(undef, nngp.n)
    tmpp = Vector{T}(undef, nngp.p)
    tmpr = Vector{T}(undef, nngp.r)
    NNGPXstar(nngp, tmpn, tmpp, tmpr)
end

eltype(xstar::NNGPXstar) = eltype(xstar.nngp.X)
issymmetric(xstar::NNGPXstar) = false
ishermitian(xstar::NNGPXstar) = false
isreal(xstar::NNGPXstar) = true
ndims(xstar::NNGPXstar) = 2

function size(xstar::NNGPXstar)
    xstar.nngp.n + xstar.nngp.p̃ * (xstar.nngp.n + xstar.nngp.r), 
    xstar.nngp.p + xstar.nngp.p̃ * (xstar.nngp.n + xstar.nngp.r)
end
size(xstar::NNGPXstar, n) = size(xstar::NNGPXstar)[n]

length(xstar::NNGPXstar) = prod(size(xstar))

"""

Overwrite `w` by `xstar * v`.
"""
function mul!(
    w::AbstractVector, 
    xstar::NNGPXstar, 
    v::AbstractVector)
    fill!(w, 0)
    n, p, p̃, r   = xstar.nngp.n, xstar.nngp.p, xstar.nngp.p̃, xstar.nngp.r
    X, X̃, y      = xstar.nngp.X, xstar.nngp.X̃, xstar.nngp.y
    Znr, Zrr     = xstar.nngp.Znr, xstar.nngp.Zrr
    A, D         = xstar.nngp.A, xstar.nngp.D
    δ²gp, δ²nngp = xstar.nngp.δ²gp, xstar.nngp.δ²nngp
    nnlist       = xstar.nngp.nnlist
    tmpn, tmpp, tmpr = xstar.tmpn, xstar.tmpp, xstar.tmpr
    # w[1:n] = X * v[1:p]
    copyto!(tmpp, 1, v, 1, p)    
    copyto!(w, mul!(tmpn, X, tmpp))
    for re in 1:p̃
        # first block rows
        copyto!(tmpr, 1, v, p + (re - 1) * r + 1, r)
        mul!(tmpn, Znr[re], tmpr) # tmpn = Zi * v[...]
        # second block rows
        lmul!(LowerTriangular(Zrr[re]), tmpr)
        woffset = n + (re - 1) * r
        multiplier = inv(sqrt(δ²gp[re]))
        for riter in 1:r
            w[woffset + riter] = tmpr[riter] * multiplier
        end
        # third block rows
        woffset = n + p̃ * r + (re - 1) * n # offset for w vector
        voffset = p + p̃ * r + (re - 1) * n # offset for v vector
        @inbounds for loc in 1:n
            li = v[voffset + loc]
            w[loc] += X̃[loc, re] * (tmpn[loc] + li)
            for (nni, nn) in enumerate(nnlist[loc])
                li -= v[voffset + nn] * A[re][loc][nni]
            end
            w[woffset + loc] = li / (sqrt(D[re][loc] * δ²nngp[re]))
        end
    end
    w
end

