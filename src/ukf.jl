struct WanMerweWeightingParameters <: AbstractWeightingParameters
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

struct UKFTimeUpdate{X,P,O} <: AbstractTimeUpdate
    state::X
    covariance::P
    χ::O
end

struct UKFTUIntermediate{O,R}
    weighted_P_chol::R
    χ::O
    χ_diff_x::O
end

UKFTUIntermediate(T::Type, num_x::Number) =
    UKFTUIntermediate(
        LowerTriangular(Matrix{T}(undef, num_x, num_x)),
        SigmaPoints(Vector{T}(undef, num_x), Matrix{T}(undef, num_x, num_x), Matrix{T}(undef, num_x, num_x)),
        SigmaPoints(Vector{T}(undef, num_x), Matrix{T}(undef, num_x, num_x), Matrix{T}(undef, num_x, num_x))
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

struct UKFMUIntermediate{O,T,R,P}
    𝓨::O
    innovation::Vector{T}
    innovation_covariance::Matrix{T}
    cross_covariance::Matrix{T}
    kalman_gain::Matrix{T}
    weighted_P_chol::R
    estimated_measurement::Vector{T}
    χ_diff_x::P
    𝓨_diff_y::O
    s_lu::Matrix{T}
end

function UKFMUIntermediate(T::Type, num_x::Number, num_y::Number)
    UKFMUIntermediate(
        SigmaPoints(Vector{T}(undef, num_y), Matrix{T}(undef, num_y, num_x), Matrix{T}(undef, num_y, num_x)),
        Vector{T}(undef, num_y),
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_x, num_y),
        LowerTriangular(Matrix{T}(undef, num_x, num_x)),
        Vector{T}(undef, num_y),
        PseudoSigmaPoints(LowerTriangular(Matrix{T}(undef, num_x, num_x))),
        SigmaPoints(Vector{T}(undef, num_y), Matrix{T}(undef, num_y, num_x), Matrix{T}(undef, num_y, num_x)),
        Matrix{T}(undef, num_y, num_y)
    )
end

UKFMUIntermediate(num_x::Number, num_y::Number) = UKFMUIntermediate(Float64, num_x, num_y)

sigmapoints(tu::UKFTimeUpdate) = tu.χ
sigmapoints(tu::UKFMeasurementUpdate) = tu.𝓨

lambda(weight_params::WanMerweWeightingParameters, L) = weight_params.α^2 * (L + weight_params.κ) - L

function calc_mean_weights(weight_params::WanMerweWeightingParameters, num_states)
    λ = lambda(weight_params, num_states)
    weight_0 = λ / (num_states + λ)
    weight_i = 1 / (2 * (num_states + λ))
    weight_0, weight_i
end

function calc_cov_weights(weight_params::WanMerweWeightingParameters, num_states)
    weight_0, weight_i = calc_mean_weights(weight_params, num_states)
    weight_0 + 1 - weight_params.α^2 + weight_params.β, weight_i
end

function calc_cholesky_weight(weight_params::WanMerweWeightingParameters, num_states)
    num_states + lambda(weight_params, num_states)
end

function calc_mean_weights(weight_params::MeanSetWeightingParameters, num_states)
    weight_params.ω₀, (1 - weight_params.ω₀) / (2num_states)
end

calc_cov_weights(weight_params::MeanSetWeightingParameters, num_states) =
    calc_mean_weights(weight_params, num_states)

function calc_cholesky_weight(weight_params::MeanSetWeightingParameters, num_states)
    num_states / (1 - weight_params.ω₀)
end

function calc_mean_weights(weight_params::GaussSetWeightingParameters, num_states)
    1 - num_states / weight_params.κ, 1 / (2weight_params.κ)
end

calc_cov_weights(weight_params::GaussSetWeightingParameters, num_states) =
    calc_mean_weights(weight_params, num_states)

function calc_cholesky_weight(weight_params::GaussSetWeightingParameters, num_states)
    weight_params.κ
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
    weight_params.α^2 * weight_params.κ
end

function mean(χ::AbstractSigmaPoints, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_mean_weights(weight_params, (size(χ, 2) - 1) >> 1)
    x = weight_0 ./ weight_i .* χ.x0
    _mean!(x, χ, weight_i)
end

function mean!(x, χ::AbstractSigmaPoints, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_mean_weights(weight_params, (size(χ, 2) - 1) >> 1)
    x .= weight_0 ./ weight_i .* χ.x0
    _mean!(x, χ, weight_i)
end

function _mean!(x, χ::AbstractSigmaPoints, weight_i)
    x = sumup!(x, χ.xi_P_plus)
    x = sumup!(x, χ.xi_P_minus)
    x .*= weight_i
end

function cov(χ_diff_x::AbstractSigmaPoints, noise, weight_params::AbstractWeightingParameters)
    cov(χ_diff_x, χ_diff_x, weight_params) .+ noise
end

function cov!(P, χ_diff_x::AbstractSigmaPoints, noise, weight_params::AbstractWeightingParameters)
    P = cov!(P, χ_diff_x, χ_diff_x, weight_params)
    P .+= noise
end

function cov(χ_diff_x::PseudoSigmaPoints, 𝓨_diff_y::AbstractSigmaPoints, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_cov_weights(weight_params, (size(χ_diff_x, 2) - 1) >> 1)
    weight_i .* (χ_diff_x.xi_P_plus * 𝓨_diff_y.xi_P_plus' .+
        χ_diff_x.xi_P_minus * 𝓨_diff_y.xi_P_minus')
end

function cov!(dest, χ_diff_x::PseudoSigmaPoints, 𝓨_diff_y::AbstractSigmaPoints, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_cov_weights(weight_params, (size(χ_diff_x, 2) - 1) >> 1)
    dest .= weight_i .* (Mul(χ_diff_x.xi_P_plus, 𝓨_diff_y.xi_P_plus') .+
        Mul(χ_diff_x.xi_P_minus, 𝓨_diff_y.xi_P_minus'))
    dest
end

function cov(χ_diff_x::AbstractSigmaPoints, 𝓨_diff_y::AbstractSigmaPoints, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_cov_weights(weight_params, (size(χ_diff_x, 2) - 1) >> 1)
    dest = weight_0 .* χ_diff_x.x0 * 𝓨_diff_y.x0'
    _cov!(dest, χ_diff_x, 𝓨_diff_y, weight_i)
end

function cov!(dest, χ_diff_x::AbstractSigmaPoints, 𝓨_diff_y::AbstractSigmaPoints, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_cov_weights(weight_params, (size(χ_diff_x, 2) - 1) >> 1)
    # Once https://github.com/JuliaArrays/LazyArrays.jl/issues/27 is fixed: dest .= weight_0 .* Mul(χ_diff_x.x0, 𝓨_diff_y.x0')
    dest .= weight_0 .* χ_diff_x.x0 * 𝓨_diff_y.x0'
    _cov!(dest, χ_diff_x, 𝓨_diff_y, weight_i)
end

function _cov!(dest, χ_diff_x::AbstractSigmaPoints, 𝓨_diff_y::AbstractSigmaPoints, weight_i)
    dest .+= weight_i .* (Mul(χ_diff_x.xi_P_plus, 𝓨_diff_y.xi_P_plus') .+
        Mul(χ_diff_x.xi_P_minus, 𝓨_diff_y.xi_P_minus'))
end

function calc_lower_triangle_cholesky(mat, weight_params::AbstractWeightingParameters)
    weight = calc_cholesky_weight(weight_params, size(mat, 1))
    cholesky(mat .* weight).L
end

function calc_lower_triangle_cholesky!(dest, mat, weight_params::AbstractWeightingParameters)
    weight = calc_cholesky_weight(weight_params, size(mat, 1))
    copyto!(dest.data, mat)
    dest.data .*= weight
    cholesky!(dest.data).L
end

function apply_func_to_sigma_points(F, x, weighted_P_chol)
    χ₁ = F(x)
    χ₂ = map(F, eachcol(x .+ weighted_P_chol))
    χ₃ = map(F, eachcol(x .- weighted_P_chol))
    SigmaPoints(χ₁, reduce(hcat, χ₂), reduce(hcat, χ₃))
end

function apply_func_to_sigma_points!(χ, F!, x, weighted_P_chol)
    F!(χ.x0, x)
    foreach(F!, eachcol(χ.xi_P_plus), eachcol(x .+ weighted_P_chol))
    foreach(F!, eachcol(χ.xi_P_minus), eachcol(x .- weighted_P_chol))
    χ
end

function create_pseudo_sigmapoints(weighted_P_chol)
    PseudoSigmaPoints(weighted_P_chol)
end

function create_pseudo_sigmapoints!(χ_diff_x, weighted_P_chol)
    χ_diff_x.xi_P_plus[:,:] .= weighted_P_chol
    χ_diff_x.xi_P_minus[:,:] .= -weighted_P_chol
    χ_diff_x
end

function time_update(x, P, F::Function, Q, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0))
    weighted_P_chol = calc_lower_triangle_cholesky(P, weight_params)
    χ = apply_func_to_sigma_points(F, x, weighted_P_chol)
    x_apri = mean(χ, weight_params)
    χ_diff_x = χ .- x_apri
    P_apri = cov(χ_diff_x, Q, weight_params)
    UKFTimeUpdate(x_apri, P_apri, χ)
end

function time_update!(tu::UKFTUIntermediate, x, P, F!::Function, Q, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0))
    χ_diff_x = tu.χ_diff_x
    weighted_P_chol = calc_lower_triangle_cholesky!(tu.weighted_P_chol, P, weight_params)
    χ = apply_func_to_sigma_points!(tu.χ, F!, x, weighted_P_chol)
    x_apri = mean!(x, χ, weight_params)
    χ_diff_x .= χ .- x_apri
    P_apri = cov!(P, χ_diff_x, Q, weight_params)
    UKFTimeUpdate(x_apri, P_apri, χ)
end

function measurement_update(x, P, y, H::Function, R, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0))
    weighted_P_chol = calc_lower_triangle_cholesky(P, weight_params)
    χ_diff_x = create_pseudo_sigmapoints(weighted_P_chol)
    𝓨 = apply_func_to_sigma_points(H, x, weighted_P_chol)
    y_est = mean(𝓨, weight_params)
    𝓨_diff_y = 𝓨 .- y_est
    ỹ = y .- y_est
    S = cov(𝓨_diff_y, R, weight_params)
    Pᵪᵧ = cov(χ_diff_x, 𝓨_diff_y, weight_params)
    K = Pᵪᵧ / S
    x_post = Mul(K, ỹ) .+ x
    P_post = calc_posterior_covariance(P, Pᵪᵧ, K)
    UKFMeasurementUpdate(x_post, P_post, 𝓨, ỹ, S, K)
end

function measurement_update!(mu::UKFMUIntermediate, x, P, y, H!::Function, R, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0))
    𝓨_diff_y, ỹ = mu.𝓨_diff_y, mu.innovation
    weighted_P_chol = calc_lower_triangle_cholesky!(mu.weighted_P_chol, P, weight_params)
    χ_diff_x = create_pseudo_sigmapoints!(mu.χ_diff_x, weighted_P_chol)
    𝓨 = apply_func_to_sigma_points!(mu.𝓨, H!, x, weighted_P_chol)
    y_est = mean!(mu.estimated_measurement, 𝓨, weight_params)
    𝓨_diff_y .= 𝓨 .- y_est
    ỹ .= y .- y_est
    S = cov!(mu.innovation_covariance, 𝓨_diff_y, R, weight_params)
    Pᵪᵧ = cov!(mu.cross_covariance, χ_diff_x, 𝓨_diff_y, weight_params)
    K = calc_kalman_gain!(mu.s_lu, mu.kalman_gain, Pᵪᵧ, S)
    x_post = calc_posterior_state!(x, K, ỹ)
    P_post = calc_posterior_covariance!(P, Pᵪᵧ, K)
    UKFMeasurementUpdate(x_post, P_post, 𝓨, ỹ, S, K)
end

function sumup!(x::AbstractVector, X::AbstractMatrix)
    for i = 1:size(X, 2)
        for j = 1:length(x)
            x[j] += X[j,i]
        end
    end
    return x
end
