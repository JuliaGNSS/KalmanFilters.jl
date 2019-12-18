UKFTUIntermediate(T::Type, num_x::Number, augment) =
    UKFTUIntermediate(
        Augmented(Cholesky(Matrix{T}(undef, num_x, num_x), 'L', 0), Cholesky(Matrix{T}(undef, num_x, num_x), 'L', 0)),
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
        Augmented(Cholesky(Matrix{T}(undef, num_x, num_x), 'L', 0), Cholesky(Matrix{T}(undef, num_x, num_x), 'L', 0)),
        Vector{T}(undef, num_y),
        AugmentedPseudoSigmaPoints(Augmented(LowerTriangular(Matrix{T}(undef, num_x, num_x)), LowerTriangular(Matrix{T}(undef, num_x, num_x)))),
        AugmentedSigmaPoints(Vector{T}(undef, num_y), Matrix{T}(undef, num_y, num_x), Matrix{T}(undef, num_y, num_y), Matrix{T}(undef, num_y, num_x), Matrix{T}(undef, num_y, num_y)),
        Matrix{T}(undef, num_y, num_y)
    )
end

UKFMUIntermediate(num_x::Number, num_y::Number, augment) = UKFMUIntermediate(Float64, num_x, num_y, augment)

struct Augmented{A, B}
    P::A
    noise::B
end

struct Augment{A}
    noise::A
end

function calc_weighted_cholesky(mat::Augmented, weight_params::AbstractWeightingParameters)
    weight = calc_cholesky_weight(weight_params, size(mat.P, 1) + size(mat.noise, 1))
    cholP = cholesky(Hermitian(mat.P .* weight))
    cholN = cholesky(mat.noise .* weight)
    Augmented(cholP, cholN)
end

function calc_weighted_cholesky!(dest::Augmented, mat::Augmented, weight_params::AbstractWeightingParameters)
    weight = calc_cholesky_weight(weight_params, size(mat.P, 1) + size(mat.noise, 1))
    copyto!(dest.P.factors, mat.P)
    copyto!(dest.noise.factors, mat.noise)
    dest.P.factors .*= weight
    dest.noise.factors .*= weight
    cholP = cholesky!(Hermitian(dest.P.factors))
    cholN = cholesky!(dest.noise.factors)
    Augmented(cholP, cholN)
end

function apply_func_to_sigma_points(F, x, weighted_chol::Augmented)
    χ₁ = F(x)
    χ₂ = map(F, eachcol(x .+ weighted_chol.P.L))
    χ₃ = map(noise -> F(x, noise), eachcol(weighted_chol.noise.L))
    χ₄ = map(F, eachcol(x .- weighted_chol.P.L))
    χ₅ = map(noise -> F(x, noise), eachcol(-weighted_chol.noise.L))
    AugmentedSigmaPoints(χ₁, reduce(hcat, χ₂), reduce(hcat, χ₃), reduce(hcat, χ₄), reduce(hcat, χ₅))
end

function apply_func_to_sigma_points!(χ, F!, x, weighted_chol::Augmented)
    F!(χ.x0, x)
    foreach(F!, eachcol(χ.xi_P_plus), eachcol(x .+ weighted_chol.P.L))
    foreach(F!, eachcol(χ.xi_P_minus), eachcol(x .- weighted_chol.P.L))
    foreach((y, noise) -> F!(y, x, noise), eachcol(χ.xi_noise_plus), eachcol(weighted_chol.noise.L))
    foreach((y, noise) -> F!(y, x, noise), eachcol(χ.xi_noise_minus), eachcol(-weighted_chol.noise.L))
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
    weight_0, weight_i = calc_cov_weights(weight_params, (size(𝓨_diff_y, 2) - 1) >> 1)
    weight_i .* (χ_diff_x.xi_P_plus * 𝓨_diff_y.xi_P_plus' .+
        χ_diff_x.xi_P_minus * 𝓨_diff_y.xi_P_minus')
end

function cov!(dest, χ_diff_x::AugmentedPseudoSigmaPoints, 𝓨_diff_y::AbstractSigmaPoints, weight_params::AbstractWeightingParameters)
    weight_0, weight_i = calc_cov_weights(weight_params, (size(𝓨_diff_y, 2) - 1) >> 1)
    dest .= weight_i .* (χ_diff_x.xi_P_plus * 𝓨_diff_y.xi_P_plus' .+
        χ_diff_x.xi_P_minus * 𝓨_diff_y.xi_P_minus')
end

function _cov!(dest, χ_diff_x::AugmentedSigmaPoints, 𝓨_diff_y::AugmentedSigmaPoints, weight_i)
    dest .+= weight_i .* (
        χ_diff_x.xi_P_plus * 𝓨_diff_y.xi_P_plus' .+
        χ_diff_x.xi_P_minus * 𝓨_diff_y.xi_P_minus' .+
        χ_diff_x.xi_noise_plus * 𝓨_diff_y.xi_noise_plus' .+
        χ_diff_x.xi_noise_minus * 𝓨_diff_y.xi_noise_minus'
    )
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
    AugmentedPseudoSigmaPoints(weighted_P_chol.P.L, -weighted_P_chol.P.L)
end

function create_pseudo_sigmapoints!(χ_diff_x, weighted_P_chol::Augmented)
    χ_diff_x.xi_P_plus .= weighted_P_chol.P.L
    χ_diff_x.xi_P_minus .= -weighted_P_chol.P.L
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
