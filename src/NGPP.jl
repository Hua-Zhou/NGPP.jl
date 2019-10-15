__precompile__()

module NGPP

using Distances, LinearAlgebra, SparseArrays, SpecialFunctions
import Base: length, size
import LinearAlgebra: BlasReal, copytri!, mul!
import SparseArrays: sparse

export
    # types
    NNGP,
    NNGPXstar,
    # functions
    matern,
    matern!,
    sweep!,
    update_AD!,
    update_Z!

"""
    NNGP

TODO
"""
struct NNGP{T<:AbstractFloat}
    # Dimensions
    "`m`: number of nearest neighbors"
    m::Int
    "`n`: number of spatial locations"
    n::Int
    "`p`: number of covariates including intercept"
    p::Int
    "`p̃`: number of covariates with varying coefficients"
    p̃::Int
    "`r`: number of knots"
    r::Int
    # Data
    "`X`: regular covariates, `n`-by-`p`"
    X::Matrix{T}
    "`X̃`: covariates with varying coefficients, `n`-by-`p̃`"
    X̃::Matrix{T}
    "`y`: response variables, `n` vector"
    y::Vector{T}
    "`coord`: coordinates of `n` locations, `n`-by-`2`"
    coord::Matrix{T}
    "`nnlist`: `nnlist[i]` lists locations of neighbors of loc `i`"
    nnlist::Vector{Vector{Int}}
    "`nndist`: `nndist[i]` is distance matrix between loc `i` and its neighbors"
    nndist::Vector{Matrix{T}}
    "`Dgpknot`: `r`-by-`r`"
    Dgpknot::Matrix{T}
    "`Dgptall`: `n`-by-`r`"
    Dgptall::Matrix{T}
    # Parameters
    "`β`: regression coefficients, `p` vector"
    β::Vector{T}
    "`δ²gpp`: variance parameters, `p̃` vector"
    δ²gp::Vector{T}
    "`δ²nngp`: variance parameters, `p̃` vector"
    δ²nngp::Vector{T}
    "`ϕgpp`: variance parameters, `p̃` vector"
    ϕgp::Vector{T}
    "`ϕnnpp`: variance parameters, `p̃` vector"
    ϕnngp::Vector{T}
    # Working arrays
    "`A`: `A[i]` is `m`-by-`n` matrix"
    A::Vector{Vector{Vector{T}}}
    "`D`: `D[i]` is `n` vector"
    D⁻½::Vector{Vector{T}}
    "`Znr`: `Z[i]` is `n`-by-`r` matrix"
    Znr::Vector{Matrix{T}}
    "`Zrr`: `Z[i]` is `r`-by-`r` matrix"
    Zrr::Vector{Matrix{T}}
    "`Cnn`: (m+1)-by-(m+1) matrix"
    Cnn::Matrix{T}
    "`Cgpknot`: r-by-r matrix"
    Cknot::Vector{Matrix{T}}
    "`Cgptall`: n-by-r matrix"
    Ctall::Vector{Matrix{T}}
end

# constructor
function NNGP(
    X::AbstractMatrix{T},
    X̃::AbstractMatrix{T},
    y::Vector{T},
    coord::Matrix{T},
    r::Integer, # number of knots
    m::Integer, # number of nearest neighbors
    nnidx::Vector{<:Integer},
    nnlu::Vector{<:Integer}, # nnidx[nnlu[i]:nnlu[i+1]-1] are the neighbors of location i
    Dgpknot::Matrix{T},
    Dgptall::Matrix{T}
    ) where T<:AbstractFloat
    n, p, p̃ = size(X, 1), size(X, 2), size(X̃, 2)
    @assert length(nnlu) == n + 1 "length of nnlu should be n+1"
    nnlist = [Vector{Int}(undef, nnlu[loc + 1] - nnlu[loc]) for loc in 1:n]
    nndist = [Matrix{T}(undef, length(nnlist[loc])+1, length(nnlist[loc])+1) for loc in 1:n]
    for loc in 1:n
        start, stop = nnlu[loc], nnlu[loc + 1] - 1
        nnlist[loc][:] = nnidx[start:stop]
        @views pairwise!(nndist[loc], Euclidean(), coord[[nnlist[loc]..., loc], :], dims=1)
    end
    β      = Vector{T}(undef, p)
    δ²gp   = Vector{T}(undef, p̃)
    δ²nngp = Vector{T}(undef, p̃)
    ϕgp    = Vector{T}(undef, p̃)
    ϕnngp  = Vector{T}(undef, p̃)
    A      = [[Vector{T}(undef, length(nnlist[loc])) for loc in 1:n] for _ in 1:p̃]
    D⁻½    = [Vector{T}(undef, n) for _ in 1:p̃]
    Znr    = [Matrix{T}(undef, n, r) for _ in 1:p̃]
    Zrr    = [Matrix{T}(undef, r, r) for _ in 1:p̃]
    Cnn    = Matrix{T}(undef, m + 1, m + 1)
    Cknot  = [Matrix{T}(undef, r, r) for _ in 1:p̃]
    Ctall  = [Matrix{T}(undef, n, r) for _ in 1:p̃]
    return NNGP{T}(m, n, p, p̃, r,
        X, X̃, y, coord, nnlist, nndist, Dgpknot, Dgptall,
        β, δ²gp, δ²nngp, ϕgp, ϕnngp,
        A, D⁻½, Znr, Zrr, Cnn, Cknot, Ctall)
end

"""
    update_AD(nngp)

Update the matrix `A` and `D` according to parameter values.
"""
function update_AD!(nngp::NNGP)
    @inbounds for re in 1:nngp.p̃
        for loc in 1:nngp.n
            nnn = length(nngp.nnlist[loc]) # number of nearest neighbors
            if nnn == 0
                nngp.D⁻½[re][loc] = 1
                continue
            end
            # evaluate Matern kernel at upper triangular part of C
            for j in 1:nnn+1, i in 1:j
                nngp.Cnn[i, j] = matern(nngp.nndist[loc][i, j], 1//2, nngp.ϕnngp[re])
            end
            # sweep the nearest neighbor Matern kernel
            sweep!(nngp.Cnn, 1:nnn, nnn + 1)
            for i in 1:nnn
                nngp.A[re][loc][i] = nngp.Cnn[i, nnn+1]
            end
            nngp.D⁻½[re][loc] = inv(sqrt(nngp.Cnn[nnn+1, nnn+1]))
        end
    end
    nothing
end

function update_Z!(nngp::NNGP)
    for re in 1:nngp.p̃
        @inbounds  for j in 1:nngp.r, i in 1:nngp.n
            nngp.Znr[re][i, j] = matern(nngp.Dgptall[i, j], 1//2, nngp.ϕgp[re])
        end
        @inbounds for j in 1:nngp.r, i in 1:j
            nngp.Zrr[re][i, j] = matern(nngp.Dgpknot[i, j], 1//2, nngp.ϕgp[re])
        end
        copytri!(nngp.Zrr[re], 'U')
        # upper triangle of Zrr stores the cholseky U factor now
        cholesky!(nngp.Zrr[re], Val(false))        
        # lower triangle of Zrr stores inv(L⋆)
        # upper triangle of Zrr stores inv(U⋆) = inv(L⋆)'
        nngp.Zrr[re][:] = inv(UpperTriangular(nngp.Zrr[re]))
        copytri!(nngp.Zrr[re], 'U')
        # Znr = Znr * inv(Zrr)
        rmul!(nngp.Znr[re], UpperTriangular(nngp.Zrr[re]))
        rmul!(nngp.Znr[re], LowerTriangular(nngp.Zrr[re]))
    end
    nothing
end

include("matern.jl")
include("xstar.jl")
include("sparse.jl")

end # module