abstract type AbstractWeightingParameters end

struct WanMerveWeightingParameters <: AbstractWeightingParameters
    α::Float64
    β::Float64
    κ::Float64
end

struct MeanSetWeightingParameters <: AbstractWeightingParameters
    ω₀::Float64
end

struct GaussSetWeightingParameters <: AbstractWeightingParameters
    κ::Float64
end

struct ScaledSetWeightingParameters <: AbstractWeightingParameters
    α::Float64
    β::Float64
    κ::Float64
end

struct UKFTimeUpdate{X,P,T} <: AbstractTimeUpdate
    state::X
    covariance::P
    χ::SigmaPoints{T}
end

struct UKFTUIntermediate{T, R <: Union{Augment{T}, <:AbstractArray{T, 2}}}
    weighted_P_chol_data::R
    χ::SigmaPoints{T}
    χ_diff_x::SigmaPoints{T}
end

struct UKFMeasurementUpdate{X,P,T} <: AbstractMeasurementUpdate
    state::X
    covariance::P
    𝓨::SigmaPoints{T}
    innovation::Vector{T}
    innovation_covariance::Matrix{T}
    cross_covariance::Matrix{T}
    kalman_gain::Matrix{T}
end

struct UKFMUIntermediate{T, R <: Union{Augment{T}, <:AbstractArray{T, 2}}}
    𝓨::SigmaPoints{T}
    innovation::Vector{T}
    innovation_covariance::Matrix{T}
    cross_covariance::Matrix{T}
    kalman_gain::Matrix{T}
    weighted_P_chol_data::R
    estimated_measurement::Vector{T}
    χ_diff_x::PseudoSigmaPoints{T}
    𝓨_diff_y::SigmaPoints{T}
    S_lu::Matrix{T}
end

sigmapoints(tu::UKFTimeUpdate) = tu.χ
sigmapoints(tu::UKFMeasurementUpdate) = tu.𝓨

lambda(weight_params::WanMerveWeightingParameters, L) = weight_params.α^2 * (L + weight_params.κ) - L

function calc_mean_weights(weight_params::WanMerveWeightingParameters, num_states)
    λ = lambda(weight_params, num_states)
    weight_0 = λ / (num_states + λ)
    weight_i = 1 / (2 * (num_states + λ))
    weight_0, weight_i
end

function calc_cov_weights(weight_params::WanMerveWeightingParameters, num_states)
    weight_0, weight_i = calc_mean_weights(weight_params, num_states)
    weight_0 + 1 - weight_params.α^2 + weight_params.β, weight_i
end

function calc_cholesky_weight(weight_params::WanMerveWeightingParameters, num_states)
    sqrt(num_states + lambda(weight_params, num_states))
end

function calc_mean_weights(weight_params::MeanSetWeightingParameters, num_states)
    weight_params.ω₀, (1 - weight_params.ω₀) / (2num_states)
end

calc_cov_weights(weight_params::MeanSetWeightingParameters, num_states) =
    calc_mean_weights(weight_params, num_states)

function calc_cholesky_weight(weight_params::MeanSetWeightingParameters, num_states)
    sqrt(num_states / (1 - weight_params.ω₀))
end

function calc_mean_weights(weight_params::GaussSetWeightingParameters, num_states)
    1 - num_states / weight_params.κ, 1 / (2weight_params.κ)
end

calc_cov_weights(weight_params::GaussSetWeightingParameters, num_states) =
    calc_mean_weights(weight_params, num_states)

function calc_cholesky_weight(weight_params::GaussSetWeightingParameters, num_states)
    sqrt(weight_params.κ)
end

function calc_mean_weights(weight_params::ScaledSetWeightingParameters, num_states)
    weight_0 = (weight_params.α^2 * weight_params.κ - num_states) / (weight_params.α^2 * weight_params.κ)
    weight_i = 1 / (2 * weight_params.α^2 * weight_params.κ)
    weight_0, weight_i
end

function calc_cov_weights(weight_params::ScaledSetWeightingParameters, num_states)
    weight_0, weight_i = calc_mean_weights(weight_params, num_states)
    weight_0 + 1 - weight_params.α^2 + weight_params.β, weight_i
end

function calc_cholesky_weight(weight_params::ScaledSetWeightingParameters, num_states)
    weight_params.α * sqrt(weight_params.κ)
end

function weighted_mean!(x, χ, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_mean_weights(weight_params, size(χ, 1))
    x .= weight_0 ./ weight_i .* χ.x0
    _weighted_mean!(x, χ, weight_i)
end

function weighted_mean(χ, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_mean_weights(weight_params, size(χ, 1))
    x = weight_0 ./ weight_i .* χ.x0
    _weighted_mean!(x, χ, weight_i)
end

function _weighted_mean!(x, χ, weight_i)
    x = sum!(x, χ.xi_plus)
    x = sum!(x, χ.xi_minus)
    x .*= weight_i
end

function weighted_cov!(P, χ_diff_x, noise, weight_params::AbstractWeightingParameters)
    P .= weighted_cross_cov!(P, χ_diff_x, χ_diff_x, weight_params) .+ noise
end

function weighted_cov(χ_diff_x, noise, weight_params::AbstractWeightingParameters)
    weighted_cross_cov(χ_diff_x, χ_diff_x, weight_params) .+ noise
end

function weighted_cross_cov(χ_diff_x::PseudoSigmaPoints, 𝓨_diff_y, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_cov_weights(weight_params, (size(χ_diff_x, 2) - 1) >> 1)
    weight_i .* (Mul(χ_diff_x.xi_P_plus, 𝓨_diff_y.xi_P_plus') .+
        Mul(χ_diff_x.xi_P_minus, 𝓨_diff_y.xi_P_minus'))
end

function weighted_cross_cov!(χ_diff_x::PseudoSigmaPoints, 𝓨_diff_y, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_cov_weights(weight_params, (size(χ_diff_x, 2) - 1) >> 1)
    dest .= weight_i .* (Mul(χ_diff_x.xi_P_plus, 𝓨_diff_y.xi_P_plus') .+
        Mul(χ_diff_x.xi_P_minus, 𝓨_diff_y.xi_P_minus'))
end

function weighted_cross_cov(χ_diff_x, 𝓨_diff_y, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_cov_weights(weight_params, (size(χ_diff_x, 2) - 1) >> 1)
    dest = weight_0 ./ weight_i .* Mul(χ_diff_x.x0, 𝓨_diff_y.x0')
    _weighted_cross_cov!(dest, χ_diff_x, 𝓨_diff_y, weight_i)
end

function weighted_cross_cov!(dest, χ_diff_x, 𝓨_diff_y, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_cov_weights(weight_params, (size(χ_diff_x, 2) - 1) >> 1)
    dest .= weight_0 ./ weight_i .* Mul(χ_diff_x.x0, 𝓨_diff_y.x0')
    _weighted_cross_cov!(dest, χ_diff_x, 𝓨_diff_y, weight_i)
end

function _weighted_cross_cov!(dest, χ_diff_x, 𝓨_diff_y, weight_i)
    dest .= weight_i .* (Mul(χ_diff_x.xi_P_plus, 𝓨_diff_y.xi_P_plus') .+
        Mul(χ_diff_x.xi_P_minus, 𝓨_diff_y.xi_P_minus'))
end

function calc_weighted_lower_triangle_cholesky(mat, weight_params::AbstractWeightingParameters)
    weight = calc_cholesky_weight(size(mat, 1), weight_params)
    cholesky(mat).L .* weight
end

function calc_weighted_lower_triangle_cholesky!(dest, mat, weight_params::AbstractWeightingParameters)
    weight = calc_cholesky_weight(size(mat, 1), weight_params)
    copyto!(dest.data, mat)
    mat_chol = cholesky!(dest.data).L
    mat_chol .*= weight
end

function apply_func_to_sigma_points(F, x, weighted_P_chol)
    χ₁ = F(x)
    χ₂ = map(F, eachcol(x .+ weighted_P_chol))
    χ₃ = map(F, eachcol(x .- weighted_P_chol))
    SigmaPoints(χ₁, reduce(hcat, χ₂), reduce(hcat, χ₃))
end

function apply_func_to_sigma_points!(χ, F!, x, weighted_P_chol)
    F!(χ.x0, x)
    foreach(F!, eachcol(χ.xi_plus), eachcol(x .+ weighted_P_chol))
    foreach(F!, eachcol(χ.xi_minus), eachcol(x .- weighted_P_chol))
    χ
end

function time_update(mu::T, F::Function, Q, weight_params::AbstractWeightingParameters = WanMerveWeightingParameters(1e-3, 2, 0)) where T <: Union{KalmanInits, <:AbstractMeasurementUpdate}
    x, P = state(mu), covariance(mu)
    weighted_P_chol = calc_weighted_lower_triangle_cholesky(P, weight_params)
    χ = apply_func_to_sigma_points(F, x, weighted_P_chol)
    x_apri = weighted_mean(χ, weight_params)
    χ_diff_x = χ .- x_apri
    P_apri = weighted_cov(χ_diff_x, Q, weight_params)
    UKFTimeUpdate(x_apri, P_apri, χ)
end

function time_update!(tu::UKFTUIntermediate, mu::T, F!::Function, Q, weight_params::AbstractWeightingParameters = WanMerveWeightingParameters(1e-3, 2, 0)) where T <: Union{KalmanInits, <:AbstractMeasurementUpdate}
    x, P = state(mu), covariance(mu)
    χ_diff_x = tu.χ_diff_x
    weighted_P_chol = calc_weighted_lower_triangle_cholesky!(tu.weighted_P_chol_data, P, weight_params)
    χ = apply_func_to_sigma_points!(tu.χ, F!, x, weighted_P_chol)
    x_apri = weighted_mean!(x, χ, weight_params)
    χ_diff_x .= χ .- x_apri
    P_apri = weighted_cov!(P, χ_diff_x, Q, weight_params)
    UKFTimeUpdate(x_apri, P_apri, χ)
end

function measurement_update(y, tu::T, H::Function, R, weight_params::AbstractWeightingParameters = WanMerveWeightingParameters(1e-3, 2, 0)) where T <: Union{KalmanInits, <:AbstractTimeUpdate}
    x, P = state(tu), covariance(tu)
    weighted_P_chol = calc_weighted_lower_triangle_cholesky(P, weight_params)
    χ_diff_x = PseudoSigmaPoints(weighted_P_chol)
    𝓨 = apply_func_to_sigma_points(H, x, weighted_P_chol)
    y_est = weighted_mean(𝓨, weight_params)
    𝓨_diff_y = 𝓨 .- y_est
    ỹ = y .- y_est
    S = weighted_cov(𝓨_diff_y, R, weight_params)
    Pᵪᵧ = weighted_cross_cov(χ_diff_x, 𝓨_diff_y, weight_params)
    K = Pᵪᵧ / S
    x_post = Mul(K, ỹ) .+ x
    P_post = (-1.) .* Mul(Pᵪᵧ, K') .+ P
    UKFMeasurementUpdate(x_post, P_post, 𝓨, ỹ, S, K)
end

function measurement_update!(mu::UKFMUIntermediate, y, tu::T, H::Function, R, weight_params::AbstractWeightingParameters = WanMerveWeightingParameters(1e-3, 2, 0)) where T <: Union{KalmanInits, <:AbstractTimeUpdate}
    x, P = state(tu), covariance(tu)
    𝓨_diff_y, ỹ = mu.𝓨_diff_y, mu.innovation
    weighted_P_chol = calc_weighted_lower_triangle_cholesky!(mu.weighted_P_chol_data, P, weight_params)
    χ_diff_x = PseudoSigmaPoints!(mu.χ_diff_x, weighted_P_chol)
    𝓨 = apply_func_to_sigma_points!(mu.𝓨, H, x, weighted_P_chol)
    y_est = weighted_mean!(mu.estimated_measurement, 𝓨, weight_params)
    𝓨_diff_y .= 𝓨 .- y_est
    ỹ .= y .- y_est
    S = weighted_cov!(mu.innovation_covariance, 𝓨_diff_y, R, weight_params)
    Pᵪᵧ = weighted_cross_cov!(mu.cross_covariance, χ_diff_x, 𝓨_diff_y, weight_params)
    K = calc_kalman_gain!(mu.s_lu, mu.kalman_gain, Pᵪᵧ, S)
    x_post = calc_posterior_state!(x, K, ỹ)
    P_post = calc_posterior_covariance!(P, Pᵪᵧ, K)
    UKFMeasurementUpdate(x_post, P_post, 𝓨, ỹ, S, K)
end
