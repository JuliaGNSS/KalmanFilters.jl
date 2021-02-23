struct UKFTimeUpdate{X,P,O} <: AbstractTimeUpdate
    state::X
    covariance::P
    χ::O
end

struct UKFTUIntermediate{T,TS}
    P_chol::Matrix{T}
    xi_temp::Vector{T}
    transformed_sigma_points::TS
    unbiased_sigma_points::TS
    x_apri::Vector{T}
    p_apri::Matrix{T}
end

UKFTUIntermediate(T::Type, num_x::Number) =
    UKFTUIntermediate(
        Matrix{T}(undef, num_x, num_x),
        Vector{T}(undef, num_x),
        TransformedSigmaPoints(Vector{T}(undef, num_x), Matrix{T}(undef, num_x, 2 * num_x), MeanSetWeightingParameters(0.0)), # Weighting parameters will be reset
        TransformedSigmaPoints(Vector{T}(undef, num_x), Matrix{T}(undef, num_x, 2 * num_x), MeanSetWeightingParameters(0.0)),
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x)
    )

UKFTUIntermediate(num_x::Number) = UKFTUIntermediate(Float64, num_x)

struct UKFMeasurementUpdate{X,P,O,T,K} <: AbstractMeasurementUpdate
    state::X
    covariance::P
    𝓨::O
    innovation::Vector{T}
    innovation_covariance::Matrix{T}
    kalman_gain::K
end

struct UKFMUIntermediate{T,TS}
    P_chol::Matrix{T}
    xi_temp::Vector{T}
    y_est::Vector{T}
    transformed_sigma_points::TS
    unbiased_sigma_points::TS
    ỹ::Vector{T}
    innovation_covariance::Matrix{T}
    cross_covariance::Matrix{T}
    s_chol::Matrix{T}
    kalman_gain::Matrix{T}
    x_posterior::Vector{T}
    p_posterior::Matrix{T}
end

function UKFMUIntermediate(T::Type, num_x::Number, num_y::Number)
    UKFMUIntermediate(
        Matrix{T}(undef, num_x, num_x),
        Vector{T}(undef, num_x),
        Vector{T}(undef, num_y),
        TransformedSigmaPoints(Vector{T}(undef, num_y), Matrix{T}(undef, num_y, 2 * num_x), MeanSetWeightingParameters(0.0)), # Weighting parameters will be reset
        TransformedSigmaPoints(Vector{T}(undef, num_y), Matrix{T}(undef, num_y, 2 * num_x), MeanSetWeightingParameters(0.0)),
        Vector{T}(undef, num_y),
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),  
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x)
    )
end

UKFMUIntermediate(num_x::Number, num_y::Number) = UKFMUIntermediate(Float64, num_x, num_y)

sigmapoints(tu::UKFTimeUpdate) = tu.χ
sigmapoints(tu::UKFMeasurementUpdate) = tu.𝓨

function time_update(x, P, f, Q, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0))
    χₖ₋₁ = calc_sigma_points(x, P, weight_params)
    χₖ₍ₖ₋₁₎ = transform(f, χₖ₋₁)
    x_apri = mean(χₖ₍ₖ₋₁₎)
    unbiased_χₖ₍ₖ₋₁₎ = substract_mean(χₖ₍ₖ₋₁₎, x_apri)
    P_apri = cov(unbiased_χₖ₍ₖ₋₁₎, Q)
    UKFTimeUpdate(x_apri, P_apri, χₖ₍ₖ₋₁₎)
end

function time_update!(tu::UKFTUIntermediate, x, P, f!, Q, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0))
    χₖ₋₁ = calc_sigma_points!(tu.P_chol, x, P, weight_params)
    χₖ₍ₖ₋₁₎ = transform!(tu.transformed_sigma_points, tu.xi_temp, f!, χₖ₋₁)
    x_apri = mean!(tu.x_apri, χₖ₍ₖ₋₁₎)
    unbiased_χₖ₍ₖ₋₁₎ = substract_mean!(tu.unbiased_sigma_points, χₖ₍ₖ₋₁₎, x_apri)
    P_apri = cov!(tu.p_apri, unbiased_χₖ₍ₖ₋₁₎, Q)
    UKFTimeUpdate(x_apri, P_apri, χₖ₍ₖ₋₁₎)
end

function measurement_update(x, P, y, h, R, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0))
    χₖ₍ₖ₋₁₎ = calc_sigma_points(x, P, weight_params)
    𝓨 = transform(h, χₖ₍ₖ₋₁₎)
    y_est = mean(𝓨)
    unbiased_𝓨 = substract_mean(𝓨, y_est)
    S = cov(unbiased_𝓨, R)
    Pᵪᵧ = cov(χₖ₍ₖ₋₁₎, unbiased_𝓨)
    ỹ = y .- y_est
    K = Pᵪᵧ / S
    x_posterior = calc_posterior_state(x, K, ỹ)
    P_posterior = calc_posterior_covariance(P, Pᵪᵧ, K)
    UKFMeasurementUpdate(x_posterior, P_posterior, 𝓨, ỹ, S, K)
end

function measurement_update!(mu::UKFMUIntermediate, x, P, y, h!, R, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0))
    χₖ₍ₖ₋₁₎ = calc_sigma_points!(mu.P_chol, x, P, weight_params)
    𝓨 = transform!(mu.transformed_sigma_points, mu.xi_temp, h!, χₖ₍ₖ₋₁₎)
    y_est = mean!(mu.y_est, 𝓨)
    unbiased_𝓨 = substract_mean!(mu.unbiased_sigma_points, 𝓨, y_est)
    S = cov!(mu.innovation_covariance, unbiased_𝓨, R)
    Pᵪᵧ = cov!(mu.cross_covariance, χₖ₍ₖ₋₁₎, unbiased_𝓨)
    mu.ỹ .= y .- y_est
    K = calc_kalman_gain!(mu.s_chol, mu.kalman_gain, Pᵪᵧ, S)
    x_posterior = calc_posterior_state!(mu.x_posterior, x, K, mu.ỹ)
    P_posterior = calc_posterior_covariance!(mu.p_posterior, P, Pᵪᵧ, K)
    UKFMeasurementUpdate(x_posterior, P_posterior, 𝓨, mu.ỹ, S, K)
end