"""
$(SIGNATURES)

Average number of sigma bound exceedings

Returns the average number of sequence values that exceed the ⨦σ bound
"""
function mean_num_sigma_bound_exceedings(sequence_over_time, covariance_over_time)
    mean(map((𝐱, 𝐘) ->  abs.(𝐱) .> sqrt.(diag(𝐘.*ones(1,1))), sequence_over_time, covariance_over_time))
end



"""
$(SIGNATURES)

Innovation magnitude bound test (σ-bound-test)

Tests if approximately 68% of sequence values lie within the ⨦σ bound
"""
function sigma_bound_test(sequence_over_time, covariance_over_time)
    return isapprox.(mean_num_sigma_bound_exceedings(sequence_over_time, covariance_over_time), [.32], atol = .015)
end



"""
$(SIGNATURES)

Innovation magnitude bound test (2σ-bound-test)

Tests if approximately 95% of sequence values lie within the ⨦2σ bound
"""
function two_sigma_bound_test(sequence_over_time, covariance_over_time)
    return isapprox.(mean_num_sigma_bound_exceedings(sequence_over_time, covariance_over_time), [.05], atol = .008)
end



"""
$(SIGNATURES)

Normalized innovation squared (NIS) Test

Double-tailed siginicance test with false alarm probability α = 0.05

Calculates confidence interval [r1 r2] and tests Prob{ ∑ NIS values)} ∈ [r1 r2] ∣ H_0 ) = 1 - α
with Hypothesis H_0: N * ∑ NIS values ∼ χ^2_{dof}
     dof (degree of freedom): N * m (N: window length, m: dimension of sequence vector)
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

Returns NIS-value for a single innovation sequence (seq) and its variance (var)
"""
function nis(seq, var)
    dot(seq, var \ seq)
end
