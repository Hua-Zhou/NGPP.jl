"""
matern(d, ϕ, ν, σ²)

Evaluate Matern kernel at distance value `d`.
"""
function matern(d::Real, ν::Real=1//2, ϕ::Real=1, σ²::Real=1)
    r = sqrt(2ν) * d * ϕ
    if ν == 1//2
        c = exp(-r)
    elseif ν == 3//2
        a = sqrt(3) * r
        c = (1 + a) * exp(-a)
    elseif ν == 5//2
        a = sqrt(5) * r
        c = (1 + a + abs2(a) / 3) * exp(- a)
    elseif ν == typemax(typeof(ν))
        c = exp((-1//2) * abs2(d * ϕ))
    else
        l = (1 - ν) * log(2) + ν * log(r) - (logabsgamma(ν))[1]
        c = exp(l) * besselk(ν, r)
    end
    σ² * c
end

function matern!(D::AbstractMatrix{<:Real}, ν::Real=1//2, ϕ::Real=1, σ²::Real=1; uplo::Char='U')
    if uplo == 'U'
        for j in 1:size(D, 2), i in 1:j
            D[i, j] = matern(D[i, j], ν, ϕ, σ²)
        end
    elseif uplo == 'L'
        for j in 1:size(D, 2), i in j:size(D, 1)
            D[i, j] = matern(D[i, j], ν, ϕ, σ²)
        end
    else
        for (i, d) in enumerate(D)
            D[i] = matern(d, ν, ϕ, σ²)
        end    
    end
    D
end

"""
Only upper triangular part is read and modified.
"""
function sweep!(A::AbstractMatrix, k::Integer, p::Integer=size(A, 2); inv::Bool=false)
    piv = 1 / A[k, k] # pivot
    # update entries other than k-th row and column
    @inbounds for j in 1:p
        j == k && continue
        akjpiv = j > k ? A[k, j] * piv : A[j, k] * piv
        for i in 1:j
            i == k && continue
            aik = i > k ? A[k, i] : A[i, k]
            A[i, j] -= aik * akjpiv
        end
    end
    # update entries other than k-th row and column
    multiplier = inv ? -piv : piv
    @inbounds for i in 1:k-1
        A[i, k] *= multiplier
    end
    @inbounds for j in k+1:p
        A[k, j] *= multiplier
    end
    # update (k, k)-entry
    @inbounds A[k, k] = -piv
    A
end

function sweep!(
    A::AbstractMatrix, 
    ks::AbstractVector{<:Integer}, 
    p::Integer=size(A, 2);
    inv::Bool=false)
    for k in ks
        sweep!(A, k, p, inv=inv)
    end
    A
end

