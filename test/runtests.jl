using BenchmarkTools, Distances, InteractiveUtils, IterativeSolvers, 
    LinearAlgebra, LinearMaps, Profile, SparseArrays, Test
using NGPP, RData, FileIO

@testset "Matern" begin
x = 3
# @code_warntype matern(x, 1//2, 0.5, 1.0)
@test matern(x) == exp(-x)
@test matern(x, Inf) == exp(- abs2(x) / 2)
@test matern.(1:2, [0.5, Inf]) == [exp(-1), exp(-2)]
# dot operation
n = 100
coord = randn(n, 2)
D = pairwise(Euclidean(), coord, dims=1)
@btime matern!($D, 1//2, 0.5)
end

@testset "sweep" begin
n = 100
A = randn(n, n)
AtA = A'A
M = deepcopy(AtA)
sweep!(M, 1:n-1)
@test M[1:n-1, n] ≈ AtA[1:n-1, 1:n-1] \ AtA[1:n-1, n]
@test M[n, n] ≈ AtA[n, n] - dot(AtA[1:n-1, n], AtA[1:n-1, 1:n-1] \ AtA[1:n-1, n])
sweep!(M, n)
@test UpperTriangular(M) ≈ - UpperTriangular(inv(AtA))
sweep!(M, 1:n, inv=true)
@test M ≈ AtA
@btime sweep!($M, 1:$n-1) setup=(copyto!($M, $AtA))
end

@testset "constructor" begin
data_cleaned = load("data_cleaned_small_expanded.RData", convert = true)["data_cleaned_small"]
n = size(data_cleaned, 1)
y = data_cleaned[:, :GPP]
X = [ones(n) data_cleaned[:, :LE]]
coord = convert(Matrix, data_cleaned[:, 3:4])
jlddata = load("MAPtest.jld")
nn = jlddata["NN"]
Dgpknot = jlddata["D_gpp_knots"]
Dgptall = jlddata["D_gpp_tall"]
m, r, p̃ = 5, 9, 2
nngp = NNGP(X, X, y, coord, r, m, nn.nnIndx, nn.nnIndxLU, Dgpknot, Dgptall)
fill!(nngp.ϕgp, 5)
fill!(nngp.ϕnngp, 5)
fill!(nngp.δ²gp, 5)
fill!(nngp.δ²nngp, 5)
update_AD!(nngp)
# @code_warntype update_AD!(nngp)
# @btime update_AD!($nngp) seconds=15
# for loc in 1:6
#     @show nngp.nnlist[loc]
#     @show nngp.nndist[loc]
#     @show nngp.A[1][loc]
#     @show nngp.D[1][loc]
# end
update_Z!(nngp)
# @code_warntype update_Z!(nngp)
# @btime update_Z!($nngp) seconds=15
# @show nngp.Znr[1][1:5, :]
# @show nngp.Znr[2][1:5, :]
# @show nngp.Zrr[1]
# @show nngp.Zrr[2]
xstar = NNGPXstar(nngp)
@show Base.summarysize(xstar)
@show Base.summarysize(nngp)

@info "Xstar * v"
v, w = Float64.(1:size(xstar, 2)), zeros(size(xstar, 1))
@btime mul!($w, $xstar, $v)
# mul!(w, xstar, v)
# @show w[1:5]
# @show w[n+1:n+r]
# @show w[n+r+1:n+2r]
# @show w[end-4:end]
# @show sum(w)

@info "Xstar' * v"
v, w = Float64.(1:size(xstar, 1)), zeros(size(xstar, 2))
@btime mul!($w, transpose($xstar), $v)
# mul!(w, transpose(xstar), v)
# @show w[1:5]
# @show w[n+1:n+r]
# @show w[n+r+1:n+2r]
# @show w[end-4:end]
# @show sum(w)

@info "LSMR"
ystar = [  y; zeros(p̃ * (n + r))]
sol = [X \ y; zeros(p̃ * (n + r))]
@time lsmr!(sol, xstar, ystar, verbose=false)

# Profile.clear()
# @profile lsmr!(sol, xstar, ystar, verbose=false)
# Profile.print(format=:flat)
# ystar = [  y; zeros(p̃ * (n + r))]
# sol = [X \ y; zeros(p̃ * (n + r))]
# f(y, x)  = mul!(y, xstar, x)
# fc(y, x) = mul!(y, transpose(xstar), x)
# D = LinearMap{Float64}(f, fc, size(xstar, 1), size(xstar, 2), 
#     ismutating=true, issymmetric=false)
# @time lsmr!(sol, D, ystar, verbose=true)
# ycg = xstar'ystar
# @time cg!(sol, D'D, ystar, verbose=true)

@info "convert to sparse CSC"
xstar_sp = sparse(xstar, Float32)
@show typeof(xstar_sp)
@show Base.summarysize(xstar_sp)
@show sum(xstar_sp)
ystar = [  y; zeros(p̃ * (n + r))]
sol = [X \ y; zeros(p̃ * (n + r))]
@time lsmr!(sol, xstar_sp, ystar, verbose=false)
end
