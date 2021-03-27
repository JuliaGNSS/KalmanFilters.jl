"""
$(SIGNATURES)

Average number of sigma bound exceedings

Returns the average number of unbiased states / innovation values that exceed the ⨦σ bound of the given covariance
"""
function mean_num_sigma_bound_exceedings(state_over_time::Vector{Vector{T}}, covariance_over_time::Vector{Matrix{T}}) where T
    mean(map((x, P) -> abs.(x) .> sqrt.(diag(P)), state_over_time, covariance_over_time))
end

function mean_num_sigma_bound_exceedings(state_over_time::Vector{T}, covariance_over_time::Vector{T}) where T
    mean(map((x, P) ->  abs(x) > sqrt(P), state_over_time, covariance_over_time))
end

"""
$(SIGNATURES)

Innovation magnitude bound test (σ-bound-test)

Tests if approximately 68% of state values lie within the ⨦σ bound
"""
function sigma_bound_test(state_over_time, covariance_over_time)
    isapprox.(mean_num_sigma_bound_exceedings(state_over_time, covariance_over_time), .32, atol = .015)
end

"""
$(SIGNATURES)

Innovation magnitude bound test (2σ-bound-test)

Tests if approximately 95% of state values lie within the ⨦2σ bound
"""
function two_sigma_bound_test(state_over_time, covariance_over_time)
    isapprox.(mean_num_sigma_bound_exceedings(state_over_time, 4 .* covariance_over_time), .05, atol = .008)
end

"""
$(SIGNATURES)

Normalized innovation squared (NIS) Test

Double-tailed siginicance test with false alarm probability α = 0.05

Calculates confidence interval [r1 r2] and tests Prob{ ∑ NIS values)} ∈ [r1 r2] ∣ H_0 ) = 1 - α
with Hypothesis H_0: N * ∑ NIS values ∼ χ^2_{dof}
     dof (degree of freedom): N * m (N: window length, m: dimension of state vector)
"""
function nis_test(nis_over_time, dof)
    sum_of_nis = sum(nis_over_time)

    r1 = cquantile(Chisq(dof), .975)
    r2 = cquantile(Chisq(dof), .025)

    (sum_of_nis >= r1) && (sum_of_nis <= r2)
end

"""
$(SIGNATURES)

Normalized innovation squared (NIS)

Returns NIS-value for a single innovation sequence `seq` and its variance `var`
"""
function calc_nis(seq, var)
    dot(seq, var \ seq)
end

function calc_nis(mu::AbstractMeasurementUpdate)
    calc_nis(get_innovation(mu), mu.innovation_covariance)
end