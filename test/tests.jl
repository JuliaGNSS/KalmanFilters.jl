@testset "Normalized innovation squared" begin
    num_samples = 20000

    unbiased_innos_scalar = randn(num_samples) .* sqrt(2)
    unbiased_variance_scalar = ones(num_samples) .* 2
    nis_over_time_us = map((x, σ²) -> calc_nis(x, σ²), unbiased_innos_scalar, unbiased_variance_scalar)
    @test calc_nis_test(nis_over_time_us) == true

    biased_innos_scalar = randn(num_samples) .* sqrt(2) .+ 1
    biased_variance_scalar = ones(num_samples) .* 2
    nis_over_time_bs = map((x, σ²) -> calc_nis(x, σ²), biased_innos_scalar, biased_variance_scalar)
    @test calc_nis_test(nis_over_time_bs) == false

    unbiased_innos = [randn(2) .* sqrt(2) for i = 1:num_samples]
    unbiased_variance = fill(Matrix{Float64}(I, 2, 2) .* 2, num_samples)
    nis_over_time_uv = map((x, σ²) -> calc_nis(x, σ²), unbiased_innos, unbiased_variance)
    @test calc_nis_test(nis_over_time_uv, num_measurements = 2) == true

    biased_innos = [randn(2) .* sqrt(2) .+ 1 for i = 1:num_samples]
    biased_variance = fill(Matrix{Float64}(I, 2, 2) .* 2, num_samples)
    nis_over_time_bv = map((x, σ²) -> calc_nis(x, σ²), biased_innos, biased_variance)
    @test calc_nis_test(nis_over_time_bv, num_measurements = 2) == false

    A = randn(4,4)
    P = A'A
    P_chol = cholesky(P)
    x = randn(4)
    @test calc_nis(x, P) ≈ calc_nis(x, P_chol)
end

@testset "σ bound test" begin
    num_samples = 20000

    unbiased_innos_scalar = randn(num_samples) .* sqrt(2)
    unbiased_variance_scalar = ones(num_samples) .* 2
    @test calc_sigma_bound_test(unbiased_innos_scalar, unbiased_variance_scalar) == true
    @test calc_two_sigma_bound_test(unbiased_innos_scalar, unbiased_variance_scalar) == true

    biased_innos_scalar = randn(num_samples) .* sqrt(2) .+ 1
    biased_variance_scalar = ones(num_samples) .* 2
    @test calc_sigma_bound_test(biased_innos_scalar, biased_variance_scalar) == false
    @test calc_two_sigma_bound_test(biased_innos_scalar, biased_variance_scalar) == false

    unbiased_innos = randn(num_samples, 2) * sqrt(2)
    unbiased_variance = ones(num_samples, 2) .* 2
    @test calc_sigma_bound_test(unbiased_innos, unbiased_variance) == [true, true]
    @test calc_two_sigma_bound_test(unbiased_innos, unbiased_variance) == [true, true]

    biased_innos = randn(num_samples, 2) * sqrt(2) .+ 2
    biased_variance = ones(num_samples, 2) .* 2
    @test calc_sigma_bound_test(biased_innos, biased_variance) == [false, false]
    @test calc_two_sigma_bound_test(biased_innos, biased_variance) == [false, false]
end
