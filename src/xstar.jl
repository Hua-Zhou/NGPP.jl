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
iscomplex(xstar::NNGPXstar) = false
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
    n, p, p̃, r   = xstar.nngp.n, xstar.nngp.p, xstar.nngp.p̃, xstar.nngp.r
    X, X̃, y      = xstar.nngp.X, xstar.nngp.X̃, xstar.nngp.y
    Znr, Zrr     = xstar.nngp.Znr, xstar.nngp.Zrr
    A, D⁻½       = xstar.nngp.A, xstar.nngp.D⁻½
    δ²gp, δ²nngp = xstar.nngp.δ²gp, xstar.nngp.δ²nngp
    nnlist       = xstar.nngp.nnlist
    tmpn, tmpp, tmpr = xstar.tmpn, xstar.tmpp, xstar.tmpr
    # w[1:n] = X * v[1:p]
    copyto!(tmpp, 1, v, 1, p)    
    copyto!(w, mul!(tmpn, X, tmpp))
    @inbounds for re in 1:p̃
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
        multiplier = inv(sqrt(δ²nngp[re]))
        for loc in 1:n
            li = v[voffset + loc]
            w[loc] += X̃[loc, re] * (tmpn[loc] + li)
            for (nni, nn) in enumerate(nnlist[loc])
                li -= v[voffset + nn] * A[re][loc][nni]
            end
            w[woffset + loc] = li * D⁻½[re][loc] * multiplier
        end
    end
    w
end

function mul!(
    w::AbstractVector, 
    xstar_trans::Union{Transpose{T, NNGPXstar{T}}, Adjoint{T, NNGPXstar{T}}},
    v::AbstractVector) where T <: AbstractFloat
    xstar        = xstar_trans.parent
    n, p, p̃, r   = xstar.nngp.n, xstar.nngp.p, xstar.nngp.p̃, xstar.nngp.r
    X, X̃, y      = xstar.nngp.X, xstar.nngp.X̃, xstar.nngp.y
    Znr, Zrr     = xstar.nngp.Znr, xstar.nngp.Zrr
    A, D⁻½       = xstar.nngp.A, xstar.nngp.D⁻½
    δ²gp, δ²nngp = xstar.nngp.δ²gp, xstar.nngp.δ²nngp
    nnlist       = xstar.nngp.nnlist
    tmpn, tmpp, tmpr = xstar.tmpn, xstar.tmpp, xstar.tmpr
    # w[1:p] = X' * v[1:n]
    mul!(tmpp, transpose(X), copyto!(tmpn, 1, v, 1, n))
    copyto!(w, 1, tmpp, 1, p)
    @inbounds for re in 1:p̃
        # contribution from second block rows of M'
        voffset = n + (re - 1) * r
        woffset = p + (re - 1) * r
        copyto!(tmpr, 1, v, voffset + 1, r)
        lmul!(UpperTriangular(Zrr[re]), tmpr)
        multiplier = inv(sqrt(δ²gp[re]))        
        for riter in 1:r
            w[woffset + riter] = tmpr[riter] * multiplier
        end
        # third block rows
        voffset = n + p̃ * r + (re - 1) * n
        woffset = p + p̃ * r + (re - 1) * n
        for loc in 1:n # tmpn = (I - Ai') * Di^{-1/2} * v[...]
            tmpn[loc] = v[voffset + loc] * D⁻½[re][loc]
            for (nni, nn) in enumerate(nnlist[loc]) # nn < loc by NN construction
                tmpn[nn] -= tmpn[loc] * A[re][loc][nni]
            end
        end
        multiplier = inv(sqrt(δ²nngp[re]))
        for loc in 1:n
            # contribution from third block rows of M'
            w[woffset + loc]  = tmpn[loc] * multiplier
            # contribution from third block rows of Z'
            tmpn[loc] = X̃[loc, re] * v[loc] # overwrite tmpn by X̃i .* v[1:n]
            w[woffset + loc] += tmpn[loc]
        end
        # add contribution from second block rows from Z'
        woffset = p + (re - 1) * r
        mul!(tmpr, transpose(Znr[re]), tmpn)
        for riter in 1:r
            w[woffset + riter] += tmpr[riter]
        end
    end
    w
end
