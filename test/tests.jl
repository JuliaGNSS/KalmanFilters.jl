@testset "Normalized innovation squared" begin
    number_samples = 20000

    unbiased_innos_scalar = map(x -> randn(1)*sqrt(2), 1:number_samples)
    unbiased_variance_scalar = ones(number_samples,1) .* 2
    nis_over_time_us = map((𝐱, σ²) -> nis(𝐱, σ²), unbiased_innos_scalar, unbiased_variance_scalar)
    dof_us = number_samples * size(unbiased_innos_scalar[1],1)
    @test nis_test(nis_over_time_us, dof_us) == true

    biased_innos_scalar = map(x -> randn(1) * sqrt(2) .+ 1, 1:number_samples)
    biased_variance_scalar = ones(number_samples,1) .* 2
    nis_over_time_bs = map((𝐱, σ²) -> nis(𝐱, σ²), biased_innos_scalar, biased_variance_scalar)
    dof_bs = number_samples * size(biased_innos_scalar[1],1)
    @test nis_test(nis_over_time_bs, dof_bs) == false

    unbiased_innos = map(x -> randn(2)*sqrt(2), 1:number_samples)
    unbiased_variance = fill(Matrix{Float64}(I, 2, 2) .* 2, number_samples)
    nis_over_time_uv = map((𝐱, σ²) -> nis(𝐱, σ²), unbiased_innos, unbiased_variance)
    dof_uv = number_samples * size(unbiased_innos[1],1)
    @test nis_test(nis_over_time_uv, dof_uv) == true

    biased_innos = map(x -> randn(2) * sqrt(2) .+ 1, 1:number_samples)
    biased_variance = fill(Matrix{Float64}(I, 2, 2) .* 2, number_samples)
    nis_over_time_bv = map((𝐱, σ²) -> nis(𝐱, σ²), biased_innos, biased_variance)
    dof_bv = number_samples * size(biased_innos[1],1)
    @test nis_test(nis_over_time_bv, dof_bv) == false
end

@testset "σ bound test" begin
    number_samples = 20000

    unbiased_innos_scalar = map(x -> randn() * sqrt(2), 1:number_samples)
    unbiased_variance_scalar = ones(number_samples,1) .* 2
    @test sigma_bound_test(unbiased_innos_scalar[4:end], unbiased_variance_scalar[4:end]) == [true]
    @test two_sigma_bound_test(unbiased_innos_scalar[4:end], 4 .* unbiased_variance_scalar[4:end]) == [true]

    biased_innos_scalar = map(x -> randn(1) * sqrt(2) .+ 1, 1:number_samples)
    biased_variance_scalar = ones(number_samples,1) .* 2
    @test sigma_bound_test(biased_innos_scalar[4:end], biased_variance_scalar[4:end]) == [false]
    @test two_sigma_bound_test(biased_innos_scalar[4:end], 4 .* biased_variance_scalar[4:end]) == [false]

    unbiased_innos = map(x -> randn(2) * sqrt(2), 1:number_samples)
    unbiased_variance = fill(Matrix{Float64}(I, 2, 2) .* 2, number_samples)
    @test sigma_bound_test(unbiased_innos[4:end], unbiased_variance[4:end]) == [true; true]
    @test two_sigma_bound_test(unbiased_innos[4:end], 4 .* unbiased_variance[4:end]) == [true; true]

    biased_innos = map(x -> randn(2) * sqrt(2) .+ 1, 1:number_samples)
    biased_variance = fill(Matrix{Float64}(I, 2, 2) .* 2, number_samples)
    @test sigma_bound_test(biased_innos[4:end], biased_variance[4:end]) == [false; false]
    @test two_sigma_bound_test(biased_innos[4:end], 4 .* biased_variance[4:end]) == [false; false]
end
