struct Augment{A <: AbstractArray{T, 2} where T}
    P::A
    noise::A
end

function calc_weighted_lower_triangle_cholesky(mat::Augment, weight_params::AbstractWeightingParameters)
    weight = calc_cholesky_weight(size(mat, 1), weight_params)
    Augment(cholesky(mat.P).L .* weight, cholesky(mat.noise).L .* weight)
end

function calc_weighted_lower_triangle_cholesky!(dest::Augment, mat::Augment, weight_params::AbstractWeightingParameters)
    weight = calc_cholesky_weight(size(mat, 1), weight_params)
    copyto!(dest.P.data, mat.P)
    copyto!(dest.noise.data, mat.noise)
    Augment(cholesky!(dest.P.data).L .* weight, cholesky!(dest.noise.data).L .* weight)
end

function calc_cholesky_weight(P::Augment, weight_params::WanMerveWeightingParameters)
    num_states = size(P.P, 1)
    num_noise_states = size(P.noise, 1)
    num_aug_states = num_states + num_noise_states
    sqrt(num_aug_states + lambda(weight_params, num_aug_states))
end

function apply_func_to_sigma_points(F, x, weighted_chol::Augment)
    χ₁ = F(x)
    χ₂ = map(F, eachcol(x .+ weighted_chol.P))
    χ₃ = map(F, eachcol(x .+ weighted_chol.noise))
    χ₄ = map(F, eachcol(x .- weighted_chol.P))
    χ₅ = map(F, eachcol(x .- weighted_chol.noise))
    AugmentedSigmaPoints(χ₁, reduce(hcat, χ₂), reduce(hcat, χ₃), reduce(hcat, χ₄), reduce(hcat, χ₅))
end

function apply_func_to_sigma_points!(χ, F!, x, weighted_chol::Augment)
    F!(χ.x0, x)
    foreach(F!, eachcol(χ.xi_P_plus), eachcol(x .+ weighted_chol.P))
    foreach(F!, eachcol(χ.xi_P_minus), eachcol(x .- weighted_chol.P))
    foreach(F!, eachcol(χ.xi_noise_plus), eachcol(x .+ weighted_chol.noise))
    foreach(F!, eachcol(χ.xi_noise_minus), eachcol(x .- weighted_chol.noise))
    χ
end

function _weighted_mean!(x, χ::AugmentedSigmaPoints, weight_i)
    x = sum!(x, χ.xi_P_plus)
    x = sum!(x, χ.xi_P_minus)
    x = sum!(x, χ.xi_noise_plus)
    x = sum!(x, χ.xi_noise_minus)
    x .= x .* weight_i
end

function _weighted_cross_cov!(dest, χ_diff_x::AugmentedSigmaPoints, 𝓨_diff_y::AugmentedSigmaPoints, weight_i)
    dest .= weight_i .*
        (Mul(χ_diff_x.xi_P_plus, 𝓨_diff_y.xi_P_plus') .+
        Mul(χ_diff_x.xi_P_minus, 𝓨_diff_y.xi_P_minus') .+
        Mul(χ_diff_x.xi_noise_plus, 𝓨_diff_y.xi_noise_plus') .+
        Mul(χ_diff_x.xi_noise_minus, 𝓨_diff_y.xi_noise_minus'))
end

function weighted_cov(χ_diff_x::AugmentedSigmaPoints, noise, weight_params::AbstractWeightingParameters)
    weighted_cross_cov(χ_diff_x, χ_diff_x, weight_params)
end

function weighted_cov!(P, χ_diff_x::AugmentedSigmaPoints, noise, weight_params::AbstractWeightingParameters)
    weighted_cross_cov!(P, χ_diff_x, χ_diff_x, weight_params)
end
