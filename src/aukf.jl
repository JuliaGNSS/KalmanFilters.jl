function AUKFTUIntermediate(T::Type, num_x::Number)
    xi_temp = Vector{T}(undef, num_x)
    UKFTUIntermediate(
        Augmented(Matrix{T}(undef, num_x, num_x), Matrix{T}(undef, num_x, num_x)),
        Augmented(xi_temp, xi_temp),
        TransformedSigmaPoints(
            Vector{T}(undef, num_x),
            Matrix{T}(undef, num_x, 4 * num_x),
            MeanSetWeightingParameters(0.0),
        ), # Weighting parameters will be reset
        TransformedSigmaPoints(
            Vector{T}(undef, num_x),
            Matrix{T}(undef, num_x, 4 * num_x),
            MeanSetWeightingParameters(0.0),
        ),
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x),
    )
end

AUKFTUIntermediate(num_x::Number) = AUKFTUIntermediate(Float64, num_x)

function AUKFMUIntermediate(T::Type, num_x::Number, num_y::Number)
    UKFMUIntermediate(
        Augmented(Matrix{T}(undef, num_x, num_x), Matrix{T}(undef, num_y, num_y)),
        Augmented(Vector{T}(undef, num_x), Vector{T}(undef, num_y)),
        Vector{T}(undef, num_y),
        TransformedSigmaPoints(
            Vector{T}(undef, num_y),
            Matrix{T}(undef, num_y, 2 * num_x + 2 * num_y),
            MeanSetWeightingParameters(0.0),
        ), # Weighting parameters will be reset
        TransformedSigmaPoints(
            Vector{T}(undef, num_y),
            Matrix{T}(undef, num_y, 2 * num_x + 2 * num_y),
            MeanSetWeightingParameters(0.0),
        ),
        Vector{T}(undef, num_y),
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x),
    )
end

AUKFMUIntermediate(num_x::Number, num_y::Number) = AUKFMUIntermediate(Float64, num_x, num_y)

function calc_kalman_gain_and_posterior_covariance(P::Augmented, Pᵪᵧ, S, consider)
    K = calc_kalman_gain(Pᵪᵧ, S, consider)
    P_posterior = calc_posterior_covariance(P.P, Pᵪᵧ, K, consider)
    K, P_posterior
end

function calc_kalman_gain_and_posterior_covariance!(
    s_chol,
    kalman_gain,
    p_post,
    P::Augmented,
    Pᵪᵧ,
    S,
)
    K = calc_kalman_gain!(s_chol, kalman_gain, Pᵪᵧ, S)
    P_posterior = calc_posterior_covariance!(p_post, P.P, Pᵪᵧ, K)
    K, P_posterior
end

function time_update(
    x,
    P::Union{<:AbstractMatrix,<:Cholesky},
    f,
    Q::Augment;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
)
    time_update(x, Augmented(P, Q), f, Q, weight_params)
end

function time_update!(
    tu::UKFTUIntermediate,
    x,
    P::Union{<:AbstractMatrix,<:Cholesky},
    f!,
    Q::Augment;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
)
    time_update!(tu, x, Augmented(P, Q), f!, Q, weight_params)
end

function measurement_update(
    x,
    P::Union{<:AbstractMatrix,<:Cholesky},
    y,
    h,
    R::Augment;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
    consider = nothing,
)
    measurement_update(
        x,
        Augmented(P, R),
        y,
        h,
        R;
        weight_params = weight_params,
        consider = consider,
    )
end

function measurement_update!(
    mu::UKFMUIntermediate,
    x,
    P::Union{<:AbstractMatrix,<:Cholesky},
    y,
    h!,
    R::Augment;
    weight_params::AbstractWeightingParameters = WanMerweWeightingParameters(),
)
    measurement_update!(mu, x, Augmented(P, R), y, h!, R; weight_params = weight_params)
end
