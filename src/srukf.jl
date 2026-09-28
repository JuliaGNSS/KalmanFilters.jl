struct SRUKFTUIntermediate{T,X,TS,AS<:Union{Matrix{T},Augmented{Matrix{T},Matrix{T}}}}
    P_chol::AS
    xi_temp::X
    transformed_x0_temp::Vector{T}
    transformed_sigma_points::TS
    unbiased_sigma_points::TS
    qr_zeros::Vector{T}
    qr_space::Vector{T}
    qr_A::Matrix{T}
    x_apri::Vector{T}
    p_apri::Matrix{T}
end

function SRUKFTUIntermediate(::Type{T}, num_x::Number) where {T}
    xi_temp = Vector{T}(undef, num_x)
    qr_zeros = zeros(T, 3 * num_x)
    qr_A = Matrix{T}(undef, 3 * num_x, num_x)
    qr_space_length = calc_gels_working_size(qr_A, qr_zeros)
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
        qr_zeros,
        Vector{T}(undef, qr_space_length),
        qr_A,
        Vector{T}(undef, num_x),
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
    qr_zeros::Vector{T}
    qr_space::Vector{T}
    qr_A::Matrix{T}
    innovation_covariance::Matrix{T}
    cross_covariance::Matrix{T}
    kalman_gain::Matrix{T}
    x_posterior::Vector{T}
    p_posterior::Matrix{T}
end

function SRUKFMUIntermediate(::Type{T}, num_x::Number, num_y::Number) where {T}
    qr_zeros = zeros(T, 2 * num_x + num_y)
    qr_A = Matrix{T}(undef, 2 * num_x + num_y, num_y)
    qr_space_length = calc_gels_working_size(qr_A, qr_zeros)
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
        qr_zeros,
        Vector{T}(undef, qr_space_length),
        qr_A,
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_x, num_y),
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x),
    )
end

SRUKFMUIntermediate(num_x::Number, num_y::Number) =
    SRUKFMUIntermediate(Float64, num_x, num_y)

function cov(χ::TransformedSigmaPoints, noise::Cholesky)
    weight_0, weight_i = calc_cov_weights(χ.weight_params, (size(χ, 2) - 1) >> 1)
    A = vcat(sqrt(weight_i) * χ.xi', noise.uplo === 'U' ? noise.U : noise.L')
    R = calc_upper_triangular_of_qr!(A)
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
    qr_zeros,
    qr_space,
    x0_temp,
    χ::TransformedSigmaPoints,
    noise::Cholesky,
)
    weight_0, weight_i = calc_cov_weights(χ.weight_params, (size(χ, 2) - 1) >> 1)
    qr_A[1:size(χ.xi, 2), :] .= sqrt(weight_i) .* χ.xi'
    # Wrap `factors` directly: `Cholesky`'s `U`/`L` properties are type-unstable (their
    # argument is a `Matrix`/`Adjoint` union), so they allocate a wrapper on every call.
    if noise.uplo === 'U'
        qr_A[(size(χ.xi, 2)+1):end, :] = UpperTriangular(noise.factors)
    else
        qr_A[(size(χ.xi, 2)+1):end, :] = LowerTriangular(noise.factors)'
    end
    R = calc_upper_triangular_of_qr_inplace!(res, qr_A, qr_zeros, qr_space)
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
    R = calc_upper_triangular_of_qr!(A)
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
    qr_zeros,
    qr_space,
    x0_temp,
    χ::TransformedSigmaPoints,
    noise::Augment{<:Cholesky},
)
    weight_0, weight_i = calc_cov_weights(χ.weight_params, (size(χ, 2) - 1) >> 1)
    qr_A .= sqrt(weight_i) .* χ.xi'
    R = calc_upper_triangular_of_qr_inplace!(res, qr_A, qr_zeros, qr_space)
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
    P_post = reduce(
        lowrankdowndate,
        eachcol(U);
        init = Cholesky(Matrix(P.factors), P.uplo, P.info),
    )
    K, P_post
end

function calc_kalman_gain_and_posterior_covariance!(
    U,
    P_post,
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
    P_post .= P.factors
    P_chol = Cholesky(P_post, P.uplo, P.info)
    foreach(u -> lowrankdowndate!(P_chol, u), eachcol(U))
    K, P_chol
end

function calc_kalman_gain_and_posterior_covariance(
    P::Augmented{<:Cholesky},
    Pᵪᵧ,
    S::Cholesky,
    consider,
)
    calc_kalman_gain_and_posterior_covariance(P.P, Pᵪᵧ, S, consider)
end

function calc_kalman_gain_and_posterior_covariance!(
    U,
    P_post,
    P::Augmented{<:Cholesky},
    Pᵪᵧ,
    S::Cholesky,
)
    calc_kalman_gain_and_posterior_covariance!(U, P_post, P.P, Pᵪᵧ, S)
end

function time_update!(
    tu::SRUKFTUIntermediate,
    x,
    P,
    f!::F,
    Q;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
) where {F}
    χₖ₋₁ = calc_sigma_points!(tu.P_chol, x, P, weight_params)
    χₖ₍ₖ₋₁₎ = transform!(tu.transformed_sigma_points, tu.xi_temp, f!, χₖ₋₁)
    x_apri = mean!(tu.x_apri, χₖ₍ₖ₋₁₎)
    unbiased_χₖ₍ₖ₋₁₎ = substract_mean!(tu.unbiased_sigma_points, χₖ₍ₖ₋₁₎, x_apri)
    P_apri = cov!(
        tu.p_apri,
        tu.qr_A,
        tu.qr_zeros,
        tu.qr_space,
        tu.transformed_x0_temp,
        unbiased_χₖ₍ₖ₋₁₎,
        Q,
    )
    SPTimeUpdate(x_apri, P_apri, χₖ₍ₖ₋₁₎)
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
        mu.qr_zeros,
        mu.qr_space,
        mu.transformed_x0_temp,
        unbiased_𝓨,
        R,
    )
    mu.ỹ .= y .- y_est
    Pᵪᵧ = cov!(mu.cross_covariance, χₖ₍ₖ₋₁₎, unbiased_𝓨)
    K, P_posterior = calc_kalman_gain_and_posterior_covariance!(
        mu.kalman_gain,
        mu.p_posterior,
        P,
        Pᵪᵧ,
        S,
    )
    x_posterior = calc_posterior_state!(mu.x_posterior, x, K, mu.ỹ)
    SPMeasurementUpdate(x_posterior, P_posterior, 𝓨, mu.ỹ, S, K)
end
