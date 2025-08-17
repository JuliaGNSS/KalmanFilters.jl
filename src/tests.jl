"""
$(SIGNATURES)

Tests if the absolute value of the states exceed the standard deviation.
"""
function do_state_errors_exceed_stds(state_errors::T, variances::T) where {T<:AbstractArray}
    does_state_error_exceed_std.(state_errors, variances)
end

"""
$(SIGNATURES)

Tests if the absolute value of the state exceed the variance
"""
function does_state_error_exceed_std(state_error, variance)
    abs(state_error) > sqrt(variance)
end

"""
$(SIGNATURES)

Innovation magnitude bound test (σ-bound-test)

Tests if approximately 68% of state values lie within the ⨦σ bound. The parameter
dims must be the dimenstion of Monte Carlo runs or the time dimension.
"""
function calc_sigma_bound_test(
    state_errors::AbstractMatrix{T},
    variances::AbstractMatrix{T};
    dims = 1,
    atol = 0.06,
) where {T}
    mean_state_errors_exceed_variance =
        vec(mean(do_state_errors_exceed_stds(state_errors, variances); dims = dims))
    exceed_probability = 1 - (cdf(Normal(), 1) - cdf(Normal(), -1))
    isapprox.(mean_state_errors_exceed_variance, exceed_probability; atol)
end

function calc_sigma_bound_test(
    state_errors::AbstractVector{T},
    variances::AbstractVector{T};
    dims = 1,
    atol = 0.06,
) where {T}
    only(
        calc_sigma_bound_test(
            reshape(state_errors, length(state_errors), 1),
            reshape(variances, length(state_errors), 1);
            dims,
            atol,
        ),
    )
end

"""
$(SIGNATURES)

Innovation magnitude bound test (2σ-bound-test)

Tests if approximately 95% of state values lie within the ⨦2σ bound. The parameter
dims must be the dimenstion of Monte Carlo runs or the time dimension.
"""
function calc_two_sigma_bound_test(
    state_errors::AbstractMatrix{T},
    variances::AbstractMatrix{T};
    dims = 1,
    atol = 0.05,
) where {T}
    mean_state_errors_exceed_4x_variance =
        vec(mean(do_state_errors_exceed_stds(state_errors, 4 * variances); dims = dims))
    exceed_probability = 1 - (cdf(Normal(), 2) - cdf(Normal(), -2))
    isapprox.(mean_state_errors_exceed_4x_variance, exceed_probability; atol)
end

function calc_two_sigma_bound_test(
    state_errors::AbstractVector{T},
    variances::AbstractVector{T};
    dims = 1,
    atol = 0.05,
) where {T}
    only(
        calc_two_sigma_bound_test(
            reshape(state_errors, length(state_errors), 1),
            reshape(variances, length(state_errors), 1);
            dims,
            atol,
        ),
    )
end

"""
$(SIGNATURES)

Normalized innovation squared (NIS) Test

Double-tailed siginicance test with false alarm probability α = 0.05

Calculates confidence interval [r1 r2] and tests Prob{ ∑ NIS values)} ∈ [r1 r2] ∣ H_0 ) = 1 - α
with Hypothesis H_0: N * ∑ NIS values ∼ χ^2_{dof}
dof (degree of freedom): N * m (N: window length, m: dimension of state vector)
see https://cs.adelaide.edu.au/~ianr/Teaching/Estimation/LectureNotes2.pdf
"""
function calc_nis_test(nis_values::AbstractVector; num_measurements = 1)
    degrees_of_freedom = num_measurements * length(nis_values)
    sum_of_nis_vals = sum(nis_values)

    r1 = cquantile(Chisq(degrees_of_freedom), 0.975)
    r2 = cquantile(Chisq(degrees_of_freedom), 0.025)

    r1 <= sum_of_nis_vals <= r2
end

"""
$(SIGNATURES)

Normalized innovation squared (NIS)

Returns NIS-value
"""
function calc_nis(innovation, innovation_covariance)
    real(dot(innovation, innovation_covariance \ innovation))
end

function calc_nis(mu::AbstractMeasurementUpdate)
    calc_nis(get_innovation(mu), get_innovation_covariance(mu))
end

"""
$(SIGNATURES)

Auto correlation test
"""
function innovation_correlation_test(innovations::AbstractMatrix)
    correlations = abs.(ifft(fft(innovations) .* conj(fft(innovations))))
    scaled_correlations = correlations ./ correlations[1]
    maximum.(eachcol(scaled_correlations[2:end, :])) .< 0.1 # Less than 10% correlation
end

function innovation_correlation_test(innovations::AbstractVector)
    only(innovation_correlation_test(reshape(innovations, length(innovations), 1)))
end
