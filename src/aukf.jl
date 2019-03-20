struct Augmented{A <: AbstractArray{T, 2} where T}
    P::A
    noise::A
end

struct Augment{A <: AbstractArray{T, 2} where T}
    noise::A
end

function calc_lower_triangle_cholesky(mat::Augmented, weight_params::AbstractWeightingParameters)
    weight = calc_cholesky_weight(weight_params, size(mat.P, 1) + size(mat.noise, 1))
    Augmented(cholesky(mat.P .* weight).L, cholesky(mat.noise .* weight).L)
end

function calc_lower_triangle_cholesky!(dest::Augmented, mat::Augmented, weight_params::AbstractWeightingParameters)
    weight = calc_cholesky_weight(weight_params, size(mat.P, 1) + size(mat.noise, 1))
    copyto!(dest.P.data, mat.P)
    copyto!(dest.noise.data, mat.noise)
    dest.P.data .*= weight
    dest.noise.data .*= weight
    Augmented(cholesky!(dest.P.data).L, cholesky!(dest.noise.data).L)
end

function apply_func_to_sigma_points(F, x, weighted_chol::Augmented)
    χ₁ = F(x)
    χ₂ = map(F, eachcol(x .+ weighted_chol.P))
    χ₃ = map(F, eachcol(x .+ weighted_chol.noise))
    χ₄ = map(F, eachcol(x .- weighted_chol.P))
    χ₅ = map(F, eachcol(x .- weighted_chol.noise))
    AugmentedSigmaPoints(χ₁, reduce(hcat, χ₂), reduce(hcat, χ₃), reduce(hcat, χ₄), reduce(hcat, χ₅))
end

function apply_func_to_sigma_points!(χ, F!, x, weighted_chol::Augmented)
    F!(χ.x0, x)
    foreach(F!, eachcol(χ.xi_P_plus), eachcol(x .+ weighted_chol.P))
    foreach(F!, eachcol(χ.xi_P_minus), eachcol(x .- weighted_chol.P))
    foreach(F!, eachcol(χ.xi_noise_plus), eachcol(x .+ weighted_chol.noise))
    foreach(F!, eachcol(χ.xi_noise_minus), eachcol(x .- weighted_chol.noise))
    χ
end

function _mean!(x, χ::AugmentedSigmaPoints, weight_i)
    x = sumup!(x, χ.xi_P_plus)
    x = sumup!(x, χ.xi_P_minus)
    x = sumup!(x, χ.xi_noise_plus)
    x = sumup!(x, χ.xi_noise_minus)
    x .= x .* weight_i
end

function cov(χ_diff_x::AugmentedPseudoSigmaPoints, 𝓨_diff_y::AbstractSigmaPoints, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_cov_weights(weight_params, (size(χ_diff_x, 2) - 1) >> 1)
    weight_i .* (χ_diff_x.xi_P_plus * 𝓨_diff_y.xi_P_plus' .+
        χ_diff_x.xi_P_minus * 𝓨_diff_y.xi_P_minus')
end

function cov!(dest, χ_diff_x::AugmentedPseudoSigmaPoints, 𝓨_diff_y::AbstractSigmaPoints, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_cov_weights(weight_params, (size(χ_diff_x, 2) - 1) >> 1)
    dest .= weight_i .* (Mul(χ_diff_x.xi_P_plus, 𝓨_diff_y.xi_P_plus') .+
        Mul(χ_diff_x.xi_P_minus, 𝓨_diff_y.xi_P_minus'))
    dest
end

function _cov!(dest, χ_diff_x::AugmentedSigmaPoints, 𝓨_diff_y::AugmentedSigmaPoints, weight_i)
    dest .+= weight_i .*
        (Mul(χ_diff_x.xi_P_plus, 𝓨_diff_y.xi_P_plus') .+
        Mul(χ_diff_x.xi_P_minus, 𝓨_diff_y.xi_P_minus') .+
        Mul(χ_diff_x.xi_noise_plus, 𝓨_diff_y.xi_noise_plus') .+
        Mul(χ_diff_x.xi_noise_minus, 𝓨_diff_y.xi_noise_minus'))
end

function cov(χ_diff_x::AugmentedSigmaPoints, noise::Augment, weight_params::AbstractWeightingParameters)
    cov(χ_diff_x, χ_diff_x, weight_params)
end

function cov!(P, χ_diff_x::AugmentedSigmaPoints, noise::Augment, weight_params::AbstractWeightingParameters)
    cov!(P, χ_diff_x, χ_diff_x, weight_params)
end

function create_pseudo_sigmapoints(weighted_P_chol::Augmented)
    AugmentedPseudoSigmaPoints(weighted_P_chol)
end

function create_pseudo_sigmapoints!(χ_diff_x, weighted_P_chol::Augmented)
    χ_diff_x.xi_P_plus .= weighted_P_chol.P
    χ_diff_x.xi_P_minus .= -weighted_P_chol.P
    χ_diff_x
end

function time_update(mu::T, F::Function, Q::Augment, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0)) where T <: Union{KalmanInits, <:AbstractMeasurementUpdate}
    time_update(T(mu, Q), F, Q, weight_params)
end

function time_update!(tu::UKFTUIntermediate, mu::T, F!::Function, Q::Augment, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0)) where T <: Union{KalmanInits, <:AbstractMeasurementUpdate}
    time_update!(tu, T(mu, Q), F!, Q, weight_params)
end

function measurement_update(tu::T, y, H::Function, R::Augment, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0)) where T <: Union{KalmanInits, <:AbstractTimeUpdate}
    measurement_update(mu, T(tu, Q), H, R, weight_params)
end

function measurement_update!(mu::UKFMUIntermediate, tu::T, y, H!::Function, R::Augment, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0)) where T <: Union{KalmanInits, <:AbstractTimeUpdate}
    measurement_update!(mu, T(tu, Q), H!, R, weight_params)
end
