struct SRUKFTUIntermediate{T,X,TS,AS<:Union{Matrix{T},Augmented{Matrix{T},Matrix{T}}}}
    P_chol::AS
    xi_temp::X
    transformed_x0_temp::Vector{T}
    transformed_sigma_points::TS
    unbiased_sigma_points::TS
    qr_tau::Vector{T}
    qr_space::Vector{T}
    qr_A::Matrix{T}
    # Scratch for the new upper factor when `P` stores its lower one
    p_apri::Matrix{T}
end

function SRUKFTUIntermediate(::Type{T}, num_x::Number) where {T}
    xi_temp = Vector{T}(undef, num_x)
    qr_A = Matrix{T}(undef, 3 * num_x, num_x)
    qr_tau = zeros(T, calc_qr_workspace_length(qr_A))
    qr_space_length = calc_qr_workspace_length(qr_A)
    SRUKFTUIntermediate(
        Matrix{T}(undef, num_x, num_x),
        xi_temp,
        xi_temp,
        TransformedSigmaPoints(
            Vector{T}(undef, num_x),
            Matrix{T}(undef, num_x, 2 * num_x),
            MeanSetWeightingParameters(0.0),
        ), # Weighting parameters will be reset
        TransformedSigmaPoints(
            Vector{T}(undef, num_x),
            Matrix{T}(undef, num_x, 2 * num_x),
            MeanSetWeightingParameters(0.0),
        ),
        qr_tau,
        Vector{T}(undef, qr_space_length),
        qr_A,
        Matrix{T}(undef, num_x, num_x),
    )
end

SRUKFTUIntermediate(num_x::Number) = SRUKFTUIntermediate(Float64, num_x)

struct SRUKFMUIntermediate{T,X,TS,AS<:Union{Matrix{T},Augmented{Matrix{T},Matrix{T}}}}
    P_chol::AS
    xi_temp::X
    y_est::Vector{T}
    transformed_x0_temp::Vector{T}
    transformed_sigma_points::TS
    unbiased_sigma_points::TS
    ỹ::Vector{T}
    qr_tau::Vector{T}
    qr_space::Vector{T}
    qr_A::Matrix{T}
    innovation_covariance::Matrix{T}
    cross_covariance::Matrix{T}
    kalman_gain::Matrix{T}
    downdate_temp::Vector{T}
end

function SRUKFMUIntermediate(::Type{T}, num_x::Number, num_y::Number) where {T}
    qr_A = Matrix{T}(undef, 2 * num_x + num_y, num_y)
    qr_tau = zeros(T, calc_qr_workspace_length(qr_A))
    qr_space_length = calc_qr_workspace_length(qr_A)
    SRUKFMUIntermediate(
        Matrix{T}(undef, num_x, num_x),
        Vector{T}(undef, num_x),
        Vector{T}(undef, num_y),
        Vector{T}(undef, num_y),
        TransformedSigmaPoints(
            Vector{T}(undef, num_y),
            Matrix{T}(undef, num_y, 2 * num_x),
            MeanSetWeightingParameters(0.0),
        ), # Weighting parameters will be reset
        TransformedSigmaPoints(
            Vector{T}(undef, num_y),
            Matrix{T}(undef, num_y, 2 * num_x),
            MeanSetWeightingParameters(0.0),
        ),
        Vector{T}(undef, num_y),
        qr_tau,
        Vector{T}(undef, qr_space_length),
        qr_A,
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_x, num_y),
        Vector{T}(undef, num_x),
    )
end

SRUKFMUIntermediate(num_x::Number, num_y::Number) =
    SRUKFMUIntermediate(Float64, num_x, num_y)

function cov(χ::TransformedSigmaPoints, noise::Cholesky)
    weight_0, weight_i = calc_cov_weights(χ.weight_params, (size(χ, 2) - 1) >> 1)
    A = vcat(sqrt(weight_i) * χ.xi', noise.uplo === 'U' ? noise.U : noise.L')
    R = calc_upper_triangular_of_qr!(A, calc_upper_triangular_of_stacked_qr_inplace!)
    correct_cholesky_sign!(R)
    S = Cholesky(R, 'U', 0)
    if weight_0 < 0
        P = lowrankdowndate(S, sqrt(abs(weight_0)) * χ.x0)
    else
        P = lowrankupdate(S, sqrt(abs(weight_0)) * χ.x0)
    end
    P
end

# StaticArrays: static sigma points give a static factor. A regular noise factor, e.g. the
# 1 x 1 one of `cholesky(r::Number)`, is converted to the static size of the sigma points.
function cov(χ::StaticTransformedSigmaPoints, noise::Cholesky)
    num_y = Size(χ.xi)[1]
    cov(χ, Cholesky(SMatrix{num_y,num_y}(noise.factors), noise.uplo, noise.info))
end

function cov(χ::StaticTransformedSigmaPoints, noise::Cholesky{<:Any,<:SMatrix})
    weight_0, weight_i = calc_cov_weights(χ.weight_params, (size(χ, 2) - 1) >> 1)
    A = vcat(sqrt(weight_i) * χ.xi', static_upper_factor(noise))
    # Skips the zeros of the triangular noise factor
    R = correct_cholesky_sign(calc_upper_triangular_of_qr(A, size(χ.xi, 2)))
    S = Cholesky(R, 'U', 0)
    v = sqrt(abs(weight_0)) * χ.x0
    weight_0 < 0 ? lowrankdowndate_columns(S, hcat(v)) : static_lowrankupdate(S, v)
end

# The upper factor of `C` including the zeros below the diagonal
static_upper_factor(C::Cholesky{T,<:SMatrix{N,N}}) where {T,N} =
    SMatrix{N,N,T}(UpperTriangular(C.uplo === 'U' ? C.factors : C.factors'))

# StaticArrays: the rank-1 updates run on mutable copies. They don't escape, so that they
# stay on the stack.
function lowrankdowndate_columns(C::Cholesky{T,<:SMatrix{N,N}}, V::SMatrix{N}) where {T,N}
    factors = MMatrix(C.factors)
    @inline lowrankdowndate_columns!(
        Cholesky(factors, C.uplo, C.info),
        MMatrix(V),
        MVector{N,T}(undef),
    )
    Cholesky(SMatrix(factors), C.uplo, C.info)
end

function static_lowrankupdate(C::Cholesky{T,<:SMatrix{N,N}}, v::SVector{N}) where {T,N}
    factors = MMatrix(C.factors)
    lowrankupdate_factor!(factors, C.uplo, MVector(v))
    Cholesky(SMatrix(factors), C.uplo, C.info)
end

# `LinearAlgebra.lowrankupdate!` on the `factors` of a Cholesky decomposition, small enough
# to be inlined (which the static update needs for its copies not to escape). The diagonal
# of the factor is real and positive, so the Givens rotation that zeros `v[i]` is simply
# c = A[i, i] / r and s = conj(v[i]) / r with r = √(A[i, i]² + |v[i]|²).
@inline function lowrankupdate_factor!(A, uplo, v)
    n = length(v)
    if uplo === 'U'
        @inbounds for j = 1:n
            v[j] = conj(v[j])
        end
    end
    @inbounds for i = 1:n
        aii = real(A[i, i])
        r = sqrt(abs2(aii) + abs2(v[i]))
        c = aii / r
        s = conj(v[i]) / r
        A[i, i] = r
        for j = (i+1):n
            aij = uplo === 'U' ? A[i, j] : A[j, i]
            vj = v[j]
            aij_new = c * aij + s * vj
            v[j] = -s' * aij + c * vj
            if uplo === 'U'
                A[i, j] = aij_new
            else
                A[j, i] = aij_new
            end
        end
    end
    A
end

function cov!(
    res,
    qr_A,
    qr_tau,
    qr_space,
    x0_temp,
    χ::TransformedSigmaPoints,
    noise::Cholesky,
)
    weight_0, weight_i = calc_cov_weights(χ.weight_params, (size(χ, 2) - 1) >> 1)
    qr_A[1:size(χ.xi, 2), :] .= sqrt(weight_i) .* χ.xi'
    copy_upper_factor!(view(qr_A, (size(χ.xi, 2)+1):size(qr_A, 1), :), noise)
    R = calc_upper_triangular_of_stacked_qr_inplace!(res, qr_A, qr_tau, qr_space)
    correct_cholesky_sign!(R)
    S = Cholesky(R, 'U', 0)
    x0_temp .= sqrt(abs(weight_0)) .* χ.x0
    if weight_0 < 0
        lowrankdowndate!(S, x0_temp)
    else
        lowrankupdate!(S, x0_temp)
    end
    S
end

function cov(χ::TransformedSigmaPoints, noise::Augment{<:Cholesky})
    weight_0, weight_i = calc_cov_weights(χ.weight_params, (size(χ, 2) - 1) >> 1)
    A = sqrt(weight_i) * χ.xi'
    R = calc_upper_triangular_of_qr!(A, calc_upper_triangular_of_dense_qr_inplace!)
    correct_cholesky_sign!(R)
    S = Cholesky(R, 'U', 0)
    if weight_0 < 0
        P = lowrankdowndate(S, sqrt(abs(weight_0)) * χ.x0)
    else
        P = lowrankupdate(S, sqrt(abs(weight_0)) * χ.x0)
    end
    P
end

function cov!(
    res,
    qr_A,
    qr_tau,
    qr_space,
    x0_temp,
    χ::TransformedSigmaPoints,
    noise::Augment{<:Cholesky},
)
    weight_0, weight_i = calc_cov_weights(χ.weight_params, (size(χ, 2) - 1) >> 1)
    qr_A .= sqrt(weight_i) .* χ.xi'
    R = calc_upper_triangular_of_dense_qr_inplace!(res, qr_A, qr_tau, qr_space)
    correct_cholesky_sign!(R)
    S = Cholesky(R, 'U', 0)
    x0_temp .= sqrt(abs(weight_0)) .* χ.x0
    if weight_0 < 0
        lowrankdowndate!(S, x0_temp)
    else
        lowrankupdate!(S, x0_temp)
    end
    S
end

"""
    lowrankdowndate_columns!(C::Cholesky, V::AbstractMatrix, temp = similar(V, size(V, 1))) -> C

Downdates the Cholesky factorization `C` of `A` to the factorization of `A - V * V'`.
The result is the same as applying `LinearAlgebra.lowrankdowndate!` for each column of
`V`, and like `lowrankdowndate!`, it overwrites `V`.

The Givens rotation of row `i` and column `l` only depends on the rotations of the
previous row with the same column and of the previous column in the same row. Hence,
instead of downdating all rows with one column after the other, this applies all columns
to one row after the other. Every inner loop then runs over contiguous memory (a column
of `V` and a row of the factor, which is copied into `temp` for an upper factor), so it
vectorizes, and it multiplies by the inverse instead of dividing.
"""
function lowrankdowndate_columns!(
    C::Cholesky,
    V::AbstractMatrix,
    temp::AbstractVector = similar(V, size(V, 1)),
)
    A = C.factors
    n = size(A, 1)
    if size(V, 1) != n
        throw(DimensionMismatch("updating vectors must fit size of factorization"))
    end
    if C.uplo === 'U'
        length(temp) >= n ||
            throw(DimensionMismatch("temp has length $(length(temp)), but needs $n"))
        # A loop rather than `conj!`, which the static downdate couldn't inline
        @inbounds for j in eachindex(V)
            V[j] = conj(V[j])
        end
        @inbounds for i = 1:n
            for j = i:n
                temp[j] = A[i, j]
            end
            downdate_row!(temp, V, i)
            for j = i:n
                A[i, j] = temp[j]
            end
        end
    else
        @inbounds for i = 1:n
            downdate_row!(view(A, :, i), V, i)
        end
    end
    C
end

# Applies the Givens rotations of all columns of `V` to row `i` of the factor, whose
# elements `i:n` are given in `r`, and to the elements `i+1:n` of the columns of `V`.
#
# Each rotation depends on the diagonal element left by the previous one. Computing the
# new diagonal element as `c * rii` would chain a division and a square root from one
# rotation to the next. Its squared magnitude, however, is simply
# `abs2(rii) - abs2(V[i, l])`, so only that subtraction is on the chain, and the
# divisions and square roots of consecutive rotations can overlap.
@inline function downdate_row!(r, V, i)
    n = size(V, 1)
    @inbounds rii = r[i]
    d = abs2(rii)
    @inbounds for l in axes(V, 2)
        vi = V[i, l]
        d_new = d - abs2(vi)
        # Equivalent to abs2(s) > 1 in `LinearAlgebra.lowrankdowndate!`
        d_new < 0 && throw(PosDefException(i))
        # conj(vi / rii) without a complex division
        s = conj(vi) * (rii / d)
        c = sqrt(d_new / d)
        inv_c = inv(c)
        rii *= c
        d = d_new
        @simd for j = (i+1):n
            vj = V[j, l]
            rj = (r[j] - s * vj) * inv_c
            r[j] = rj
            V[j, l] = -s' * rj + c * vj
        end
    end
    @inbounds r[i] = rii
    r
end

function calc_kalman_gain_and_posterior_covariance(
    P::Cholesky,
    Pᵪᵧ,
    S::Cholesky,
    consider::Nothing,
)
    U = S.uplo === 'U' ? Pᵪᵧ / S.U : Pᵪᵧ / S.L'
    K = S.uplo === 'U' ? U / S.U' : U / S.L
    # StaticArrays doesn't support lowrankdowndate
    # see https://github.com/JuliaArrays/StaticArrays.jl/issues/930
    P_post =
        lowrankdowndate_columns!(Cholesky(Matrix(P.factors), P.uplo, P.info), Matrix(U))
    K, P_post
end

function calc_kalman_gain_and_posterior_covariance(
    P::Cholesky{<:Any,<:SMatrix},
    Pᵪᵧ::SMatrix,
    S::Cholesky{<:Any,<:SMatrix},
    consider::Nothing,
)
    S_U = UpperTriangular(static_upper_factor(S))
    U = Pᵪᵧ / S_U
    K = U / S_U'
    K, lowrankdowndate_columns(P, U)
end

# Downdates `P` in place to the posterior factor and returns the gain with it.
function calc_kalman_gain_and_posterior_covariance!(
    U,
    downdate_temp,
    P::Cholesky,
    Pᵪᵧ,
    S::Cholesky,
)
    # Wrap `factors` directly to keep the factor types concrete (see `cov!`).
    if S.uplo === 'U'
        S_U = UpperTriangular(S.factors)
        U .= rdiv!(Pᵪᵧ, S_U)
        K = rdiv!(Pᵪᵧ, S_U')
    else
        S_L = LowerTriangular(S.factors)
        U .= rdiv!(Pᵪᵧ, S_L')
        K = rdiv!(Pᵪᵧ, S_L)
    end
    lowrankdowndate_columns!(P, U, downdate_temp)
    K, P
end

function calc_kalman_gain_and_posterior_covariance(
    P::Augmented{<:Cholesky},
    Pᵪᵧ,
    S::Cholesky,
    consider,
)
    calc_kalman_gain_and_posterior_covariance(P.P, Pᵪᵧ, S, consider)
end

function time_update!(
    tu::SRUKFTUIntermediate,
    x,
    P,
    f!::F,
    Q;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
) where {F}
    # The sigma points are drawn from `x` and `P` (copied into the intermediate) before
    # either is written, so the a priori mean lands directly in `x`, and the new upper
    # factor in `P`'s own factor where it stores the upper one.
    χₖ₋₁ = calc_sigma_points!(tu.P_chol, x, P, weight_params)
    χₖ₍ₖ₋₁₎ = transform!(tu.transformed_sigma_points, tu.xi_temp, f!, χₖ₋₁)
    x_apri = mean!(x, χₖ₍ₖ₋₁₎)
    unbiased_χₖ₍ₖ₋₁₎ = substract_mean!(tu.unbiased_sigma_points, χₖ₍ₖ₋₁₎, x_apri)
    P_own = caller_covariance(P)
    R = cov!(
        upper_factor_target(P_own, tu.p_apri),
        tu.qr_A,
        tu.qr_tau,
        tu.qr_space,
        tu.transformed_x0_temp,
        unbiased_χₖ₍ₖ₋₁₎,
        Q,
    )
    store_upper_factor!(P_own, R.factors)
    SPTimeUpdate(x_apri, P_own, χₖ₍ₖ₋₁₎)
end

function measurement_update!(
    mu::SRUKFMUIntermediate,
    x,
    P,
    y,
    h!::F,
    R;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
) where {F}
    χₖ₍ₖ₋₁₎ = calc_sigma_points!(mu.P_chol, x, P, weight_params)
    𝓨 = transform!(mu.transformed_sigma_points, mu.xi_temp, h!, χₖ₍ₖ₋₁₎)
    y_est = mean!(mu.y_est, 𝓨)
    unbiased_𝓨 = substract_mean!(mu.unbiased_sigma_points, 𝓨, y_est)
    S = cov!(
        mu.innovation_covariance,
        mu.qr_A,
        mu.qr_tau,
        mu.qr_space,
        mu.transformed_x0_temp,
        unbiased_𝓨,
        R,
    )
    mu.ỹ .= y .- y_est
    Pᵪᵧ = cov!(mu.cross_covariance, χₖ₍ₖ₋₁₎, unbiased_𝓨)
    K, P_posterior = calc_kalman_gain_and_posterior_covariance!(
        mu.kalman_gain,
        mu.downdate_temp,
        caller_covariance(P),
        Pᵪᵧ,
        S,
    )
    x_posterior = calc_posterior_state!(x, K, mu.ỹ)
    SPMeasurementUpdate(x_posterior, P_posterior, 𝓨, mu.ỹ, S, K)
end
