function weight(aug::Augmented, weight_params)
    weight = calc_cholesky_weight(weight_params, size(aug.P, 1) + size(aug.noise, 1))
    chol_P = Cholesky(aug.P.factors .* sqrt(weight), aug.P.uplo, aug.P.info)
    chol_N = Cholesky(aug.noise.factors .* sqrt(weight), aug.noise.uplo, aug.noise.info)
    Augmented(chol_P, chol_N)
end

function cov(
    χ_diff_x::AugmentedSigmaPoints,
    noise::Augment{<: Cholesky},
    weight_params::AbstractWeightingParameters
)
    weight_0, weight_i = calc_cov_weights(weight_params, (size(χ_diff_x, 2) - 1) >> 1)
    Q, R = qr(hcat(
        sqrt(weight_i) * χ_diff_x.xi_P_plus,
        sqrt(weight_i) * χ_diff_x.xi_P_minus,
        sqrt(weight_i) * χ_diff_x.xi_noise_plus,
        sqrt(weight_i) * χ_diff_x.xi_noise_minus
    )')
    S = Cholesky(R, 'U', 0)
    if weight_0 < 0
        P = lowrankdowndate(S, sqrt(abs(weight_0)) * χ_diff_x.x0)
    else
        P = lowrankupdate(S, sqrt(abs(weight_0)) * χ_diff_x.x0)
    end
    P
end

function time_update(
    x,
    P::Cholesky,
    F::Function,
    Q::Augment{<: Cholesky},
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0)
)
    augmented_P = Augmented(P, Q.noise)
    weighted_P_chol = weight(augmented_P, weight_params)
    χ = apply_func_to_sigma_points(F, x, weighted_P_chol)
    x_apriori = mean(χ, weight_params)
    χ_diff_x = χ .- x_apriori
    P_apriori = cov(χ_diff_x, Q, weight_params)
    SRUKFTimeUpdate(x_apriori, P_apriori, χ)
end

function measurement_update(
    x,
    P::Cholesky,
    y,
    H::Function,
    R::Augment{<: Cholesky},
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0)
)
    augmented_P = Augmented(P, R.noise)
    weighted_P_chol = weight(augmented_P, weight_params)
    χ_diff_x = create_pseudo_sigmapoints(weighted_P_chol)
    𝓨 = apply_func_to_sigma_points(H, x, weighted_P_chol)
    y_est = mean(𝓨, weight_params)
    𝓨_diff_y = 𝓨 .- y_est
    ỹ = y .- y_est
    S = cov(𝓨_diff_y, R, weight_params)
    Pᵪᵧ = cov(χ_diff_x, 𝓨_diff_y, weight_params)
    K = Pᵪᵧ / S.U / S.L
    x_post = Mul(K, ỹ) .+ x
    U = K * S.L
    P_post = copy(P)
    for i = 1:size(U, 2)
        P_post = lowrankdowndate(P_post, U[:,i])
    end
    SRUKFMeasurementUpdate(x_post, P_post, 𝓨, ỹ, S, K)
end
