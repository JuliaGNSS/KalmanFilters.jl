@testset "Normalized innovation squared" begin
    num_samples = 20000

    unbiased_innos_scalar = randn(num_samples) .* sqrt(2)
    unbiased_variance_scalar = ones(num_samples) .* 2
    nis_over_time_us = map((x, σ²) -> nis(x, σ²), unbiased_innos_scalar, unbiased_variance_scalar)
    @test nis_test(nis_over_time_us, num_samples) == true

    biased_innos_scalar = randn(num_samples) .* sqrt(2) .+ 1
    biased_variance_scalar = ones(num_samples) .* 2
    nis_over_time_bs = map((x, σ²) -> nis(x, σ²), biased_innos_scalar, biased_variance_scalar)
    @test nis_test(nis_over_time_bs, num_samples) == false

    unbiased_innos = [randn(2) .* sqrt(2) for i = 1:num_samples]
    unbiased_variance = fill(Matrix{Float64}(I, 2, 2) .* 2, num_samples)
    nis_over_time_uv = map((x, σ²) -> nis(x, σ²), unbiased_innos, unbiased_variance)
    dof_uv = num_samples * 2
    @test nis_test(nis_over_time_uv, dof_uv) == true

    biased_innos = [randn(2) .* sqrt(2) .+ 1 for i = 1:num_samples]
    biased_variance = fill(Matrix{Float64}(I, 2, 2) .* 2, num_samples)
    nis_over_time_bv = map((x, σ²) -> nis(x, σ²), biased_innos, biased_variance)
    dof_bv = num_samples * 2
    @test nis_test(nis_over_time_bv, dof_bv) == false
end

@testset "σ bound test" begin
    num_samples = 20000

    unbiased_innos_scalar = randn(num_samples) .* sqrt(2)
    unbiased_variance_scalar = ones(num_samples) .* 2
    @test sigma_bound_test(unbiased_innos_scalar, unbiased_variance_scalar) == true
    @test two_sigma_bound_test(unbiased_innos_scalar, unbiased_variance_scalar) == true

    biased_innos_scalar = randn(num_samples) .* sqrt(2) .+ 1
    biased_variance_scalar = ones(num_samples) .* 2
    @test sigma_bound_test(biased_innos_scalar, biased_variance_scalar) == false
    @test two_sigma_bound_test(biased_innos_scalar, biased_variance_scalar) == false

    unbiased_innos = [randn(2) .* sqrt(2) for i = 1:num_samples]
    unbiased_variance = fill(Matrix{Float64}(I, 2, 2) .* 2, num_samples)
    @test sigma_bound_test(unbiased_innos, unbiased_variance) == [true; true]
    @test two_sigma_bound_test(unbiased_innos, unbiased_variance) == [true; true]

    biased_innos = [randn(2) .* sqrt(2) .+ 1 for i = 1:num_samples]
    biased_variance = fill(Matrix{Float64}(I, 2, 2) .* 2, num_samples)
    @test sigma_bound_test(biased_innos, biased_variance) == [false; false]
    @test two_sigma_bound_test(biased_innos, biased_variance) == [false; false]
end
