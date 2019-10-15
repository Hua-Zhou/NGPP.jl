"""
    sparse(xstar, T=eltype(xstar))

Generate a `SparseMatrixCSC{T, TI}` for `xstar`.
"""
function sparse(xstar::NNGPXstar, T=Base.eltype(xstar))
    n, p, p̃, r   = xstar.nngp.n, xstar.nngp.p, xstar.nngp.p̃, xstar.nngp.r
    X, X̃, y      = xstar.nngp.X, xstar.nngp.X̃, xstar.nngp.y
    Znr, Zrr     = xstar.nngp.Znr, xstar.nngp.Zrr
    A, D⁻½       = xstar.nngp.A, xstar.nngp.D⁻½
    δ²gp, δ²nngp = xstar.nngp.δ²gp, xstar.nngp.δ²nngp
    nnlist       = xstar.nngp.nnlist
    tilist = [Int8, Int16, Int32, Int64, Int128]
    TI = tilist[findfirst(t -> typemax(t) > n, tilist)]
    S = SparseMatrixCSC{T, TI}(spzeros(size(xstar)...))
    S[1:n, 1:p] = X
    Is = [vcat([fill(loc, length(nnlist[loc])) for loc in 1:n]...); 1:n]
    Js = [vcat(nnlist...); 1:n]
    @views for re in 1:p̃
        S[1:n, (p + (re - 1) * r + 1):(p + re * r)] = 
            Diagonal(X̃[:, re]) * Znr[re]
        S[1:n, (p + p̃ * r + (re - 1) * n + 1):(p + p̃ * r + re * n)] = 
            spdiagm(0 => X̃[:, re])
        S[(n + (re - 1) * r + 1):(n + re * r),
          (p + (re - 1) * r + 1):(p + re * r)] = 
            inv(sqrt(δ²gp[re])) * LowerTriangular(Zrr[re])
        Vs = [vcat(-A[re]...); fill(1, n)]
        Lsp = sparse(Is, Js, Vs, n, n)
        lmul!(Diagonal(D⁻½[re]), Lsp)
        S[(n + p̃ * r + (re - 1) * n + 1):(n + p̃ * r + re * n),
          (p + p̃ * r + (re - 1) * n + 1):(p + p̃ * r + re * n)] = 
            inv(sqrt(δ²nngp[re])) * Lsp
    end
    dropzeros!(S)
end
