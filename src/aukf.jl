UKFTUIntermediate(T::Type, num_x::Number, augment) =
    UKFTUIntermediate(
        Augmented(LowerTriangular(Matrix{T}(undef, num_x, num_x)), LowerTriangular(Matrix{T}(undef, num_x, num_x))),
        AugmentedSigmaPoints(Vector{T}(undef, num_x), Matrix{T}(undef, num_x, num_x), Matrix{T}(undef, num_x, num_x), Matrix{T}(undef, num_x, num_x), Matrix{T}(undef, num_x, num_x)),
        AugmentedSigmaPoints(Vector{T}(undef, num_x), Matrix{T}(undef, num_x, num_x), Matrix{T}(undef, num_x, num_x), Matrix{T}(undef, num_x, num_x), Matrix{T}(undef, num_x, num_x))
    )

UKFTUIntermediate(num_x::Number, augment) = UKFTUIntermediate(Float64, num_x, augment)

function UKFMUIntermediate(T::Type, num_x::Number, num_y::Number, augment)
    UKFMUIntermediate(
        AugmentedSigmaPoints(Vector{T}(undef, num_y), Matrix{T}(undef, num_y, num_y), Matrix{T}(undef, num_y, num_x), Matrix{T}(undef, num_y, num_x), Matrix{T}(undef, num_y, num_y)),
        Vector{T}(undef, num_y),
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_x, num_y),
        Augmented(LowerTriangular(Matrix{T}(undef, num_x, num_x)), LowerTriangular(Matrix{T}(undef, num_x, num_x))),
        Vector{T}(undef, num_y),
        AugmentedPseudoSigmaPoints(Augmented(LowerTriangular(Matrix{T}(undef, num_x, num_x)), LowerTriangular(Matrix{T}(undef, num_x, num_x)))),
        AugmentedSigmaPoints(Vector{T}(undef, num_y), Matrix{T}(undef, num_y, num_x), Matrix{T}(undef, num_y, num_y), Matrix{T}(undef, num_y, num_x), Matrix{T}(undef, num_y, num_y)),
        Matrix{T}(undef, num_y, num_y)
    )
end

UKFMUIntermediate(num_x::Number, num_y::Number, augment) = UKFMUIntermediate(Float64, num_x, num_y, augment)

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
    χ₃ = map(F, x, eachcol(weighted_chol.noise))
    χ₄ = map(F, eachcol(x .- weighted_chol.P))
    χ₅ = map(F, x, eachcol(-weighted_chol.noise))
    AugmentedSigmaPoints(χ₁, reduce(hcat, χ₂), reduce(hcat, χ₃), reduce(hcat, χ₄), reduce(hcat, χ₅))
end

function apply_func_to_sigma_points!(χ, F!, x, weighted_chol::Augmented)
    F!(χ.x0, x)
    foreach(F!, eachcol(χ.xi_P_plus), eachcol(x .+ weighted_chol.P))
    foreach(F!, eachcol(χ.xi_P_minus), eachcol(x .- weighted_chol.P))
    foreach(F!, eachcol(χ.xi_noise_plus), x, eachcol(weighted_chol.noise))
    foreach(F!, eachcol(χ.xi_noise_minus), x, eachcol(-weighted_chol.noise))
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

function cov(χ_diff_x::AugmentedSigmaPoints, noise::Nothing, weight_params::AbstractWeightingParameters)
    cov(χ_diff_x, χ_diff_x, weight_params)
end

function cov!(P::Augmented, χ_diff_x::AugmentedSigmaPoints, noise::Nothing, weight_params::AbstractWeightingParameters)
    cov!(P.P, χ_diff_x, χ_diff_x, weight_params)
end

function cov!(P, χ_diff_x::AugmentedSigmaPoints, noise::Nothing, weight_params::AbstractWeightingParameters)
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

function calc_posterior_covariance(P::Augmented, Pᵪᵧ, K)
    calc_posterior_covariance(P.P, Pᵪᵧ, K)
end

function calc_posterior_covariance!(P::Augmented, PHᵀ, K)
    calc_posterior_covariance!(P.P, PHᵀ, K)
end

function time_update(x, P, F::Function, Q::Augment, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0))
    time_update(x, Augmented(P, Q.noise), F, nothing, weight_params)
end

function time_update!(tu::UKFTUIntermediate, x, P, F!::Function, Q::Augment, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0))
    time_update!(tu, x, Augmented(P, Q.noise), F!, nothing, weight_params)
end

function measurement_update(x, P, y, H::Function, R::Augment, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0))
    measurement_update(x, Augmented(P, R.noise), y, H, nothing, weight_params)
end

function measurement_update!(mu::UKFMUIntermediate, x, P, y, H!::Function, R::Augment, weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(1e-3, 2, 0))
    measurement_update!(mu, x, Augmented(P, R.noise), y, H!, nothing, weight_params)
end
