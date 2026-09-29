struct UKFTUIntermediate{T,X,TS,AS<:Union{Matrix{T},Augmented{Matrix{T},Matrix{T}}}}
    P_chol::AS
    xi_temp::X
    transformed_sigma_points::TS
    unbiased_sigma_points::TS
end

UKFTUIntermediate(::Type{T}, num_x::Number) where {T} = UKFTUIntermediate(
    Matrix{T}(undef, num_x, num_x),
    Vector{T}(undef, num_x),
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
)

UKFTUIntermediate(num_x::Number) = UKFTUIntermediate(Float64, num_x)

struct UKFMUIntermediate{T,X,TS,AS<:Union{Matrix{T},Augmented{Matrix{T},Matrix{T}}}}
    P_chol::AS
    xi_temp::X
    y_est::Vector{T}
    transformed_sigma_points::TS
    unbiased_sigma_points::TS
    ỹ::Vector{T}
    innovation_covariance::Matrix{T}
    cross_covariance::Matrix{T}
    s_chol::Matrix{T}
    kalman_gain::Matrix{T}
end

function UKFMUIntermediate(::Type{T}, num_x::Number, num_y::Number) where {T}
    UKFMUIntermediate(
        Matrix{T}(undef, num_x, num_x),
        Vector{T}(undef, num_x),
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
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),
    )
end

UKFMUIntermediate(num_x::Number, num_y::Number) = UKFMUIntermediate(Float64, num_x, num_y)

sigmapoints(tu::SPTimeUpdate) = tu.χ
sigmapoints(tu::SPMeasurementUpdate) = tu.𝓨

function time_update(
    x,
    P,
    f,
    Q,
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
)
    χₖ₋₁ = calc_sigma_points(x, P, weight_params)
    χₖ₍ₖ₋₁₎ = transform(f, χₖ₋₁)
    x_apri = mean(χₖ₍ₖ₋₁₎)
    unbiased_χₖ₍ₖ₋₁₎ = substract_mean(χₖ₍ₖ₋₁₎, x_apri)
    P_apri = cov(unbiased_χₖ₍ₖ₋₁₎, Q)
    SPTimeUpdate(x_apri, P_apri, χₖ₍ₖ₋₁₎)
end

# In place: the sigma points are drawn from `x` and `P` (copied into the intermediate)
# before either is written, so the a priori mean lands directly in `x` and the a priori
# covariance in `P`.
function time_update!(
    tu::UKFTUIntermediate,
    x,
    P,
    f!::F,
    Q,
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
) where {F}
    χₖ₋₁ = calc_sigma_points!(tu.P_chol, x, P, weight_params)
    χₖ₍ₖ₋₁₎ = transform!(tu.transformed_sigma_points, tu.xi_temp, f!, χₖ₋₁)
    x_apri = mean!(x, χₖ₍ₖ₋₁₎)
    unbiased_χₖ₍ₖ₋₁₎ = substract_mean!(tu.unbiased_sigma_points, χₖ₍ₖ₋₁₎, x_apri)
    P_apri = cov!(caller_covariance(P), unbiased_χₖ₍ₖ₋₁₎, Q)
    SPTimeUpdate(x_apri, P_apri, χₖ₍ₖ₋₁₎)
end

function measurement_update(
    x,
    P,
    y,
    h,
    R;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
    consider = nothing,
)
    χₖ₍ₖ₋₁₎ = calc_sigma_points(x, P, weight_params)
    𝓨 = transform(h, χₖ₍ₖ₋₁₎)
    y_est = mean(𝓨)
    unbiased_𝓨 = substract_mean(𝓨, y_est)
    S = cov(unbiased_𝓨, R)
    ỹ = y .- y_est
    Pᵪᵧ = cov(χₖ₍ₖ₋₁₎, unbiased_𝓨)
    K, P_posterior = calc_kalman_gain_and_posterior_covariance(P, Pᵪᵧ, S, consider)
    x_posterior = calc_posterior_state(x, K, ỹ, consider)
    SPMeasurementUpdate(x_posterior, P_posterior, 𝓨, ỹ, S, K)
end

function calc_kalman_gain_and_posterior_covariance(P, Pᵪᵧ, S, consider)
    K = calc_kalman_gain(Pᵪᵧ, S, consider)
    P_posterior = calc_posterior_covariance(P, Pᵪᵧ, K, consider)
    K, P_posterior
end

function measurement_update!(
    mu::UKFMUIntermediate,
    x,
    P,
    y,
    h!::F,
    R;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0),
) where {F}
    χₖ₍ₖ₋₁₎ = calc_sigma_points!(mu.P_chol, x, P, weight_params)
    𝓨 = transform!(mu.transformed_sigma_points, mu.xi_temp, h!, χₖ₍ₖ₋₁₎)
    y_est = mean!(mu.y_est, 𝓨)
    unbiased_𝓨 = substract_mean!(mu.unbiased_sigma_points, 𝓨, y_est)
    S = cov!(mu.innovation_covariance, unbiased_𝓨, R)
    Pᵪᵧ = cov!(mu.cross_covariance, χₖ₍ₖ₋₁₎, unbiased_𝓨)
    mu.ỹ .= y .- y_est
    K = calc_kalman_gain!(mu.s_chol, mu.kalman_gain, Pᵪᵧ, S)
    P_posterior = calc_posterior_covariance!(caller_covariance(P), Pᵪᵧ, K)
    x_posterior = calc_posterior_state!(x, K, mu.ỹ)
    SPMeasurementUpdate(x_posterior, P_posterior, 𝓨, mu.ỹ, S, K)
end
