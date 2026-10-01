# With StaticArrays the allocating UKF and SR-UKF stay on the stack: they return static
# arrays and don't allocate.
function allocations_time_update(x, P, f::F, Q, weight_params) where {F}
    time_update(x, P, f, Q, weight_params)
    return @allocated time_update(x, P, f, Q, weight_params)
end

function allocations_measurement_update(x, P, y, h::F, R, weight_params) where {F}
    measurement_update(x, P, y, h, R; weight_params)
    return @allocated measurement_update(x, P, y, h, R; weight_params)
end

@testset "Static $name with $T, $num_x states, $num_y measurements and $W" for (
        name,
        as_cov,
    ) in (
        ("UKF", identity),
        ("SRUKF", P -> cholesky(Hermitian(P))),
    ),
    T in (Float64, ComplexF64),
    (num_x, num_y) in ((1, 1), (2, 2), (4, 3)),
    # The weight of the mean sigma point is negative for the former and positive for the
    # latter, so the square root filter downdates and updates its factor respectively.
    W in (WanMerweWeightingParameters, MeanSetWeightingParameters)

    random_pos_def(n) = (A = @SMatrix(randn(T, n, n)); A'A + n * I)
    x = @SVector randn(T, num_x)
    P = random_pos_def(num_x)
    Q = random_pos_def(num_x)
    R = random_pos_def(num_y)
    y = @SVector randn(T, num_y)
    F = @SMatrix randn(T, num_x, num_x)
    H = @SMatrix randn(T, num_y, num_x)
    f(x) = F * x
    h(x) = H * x
    weight_params = W()
    cov_type = name == "UKF" ? SMatrix{num_x,num_x,T} : Cholesky{T,<:SMatrix{num_x,num_x,T}}

    tu = time_update(x, P, F, Q)
    tu_static = @inferred time_update(x, as_cov(P), f, as_cov(Q), weight_params)
    @test get_state(tu_static) isa SVector{num_x,T}
    @test tu_static.covariance isa cov_type
    @test get_state(tu_static) ≈ get_state(tu)
    @test get_covariance(tu_static) ≈ get_covariance(tu)
    @test allocations_time_update(x, as_cov(P), f, as_cov(Q), weight_params) == 0

    mu = measurement_update(x, P, y, H, R)
    mu_static = @inferred measurement_update(x, as_cov(P), y, h, as_cov(R); weight_params)
    @test get_state(mu_static) isa SVector{num_x,T}
    @test mu_static.covariance isa cov_type
    @test get_kalman_gain(mu_static) isa SMatrix{num_x,num_y,T}
    @test get_innovation(mu_static) isa SVector{num_y,T}
    @test get_state(mu_static) ≈ get_state(mu)
    @test get_covariance(mu_static) ≈ get_covariance(mu)
    @test get_kalman_gain(mu_static) ≈ get_kalman_gain(mu)
    @test allocations_measurement_update(x, as_cov(P), y, h, as_cov(R), weight_params) == 0
end

@testset "Static scalar $name measurement update" for (name, as_cov) in (
    ("UKF", identity),
    ("SRUKF", cholesky),
)
    x = @SVector randn(3)
    A = @SMatrix randn(3, 3)
    P = A'A + 3I
    r = 2.0
    y = randn()
    H = (@SVector randn(3))'
    h(x) = H * x

    mu = measurement_update(x, P, y, H, r)
    P_static = name == "UKF" ? P : cholesky(P)
    mu_static = @inferred measurement_update(x, P_static, y, h, as_cov(r))
    @test get_state(mu_static) isa SVector{3,Float64}
    @test get_state(mu_static) ≈ get_state(mu)
    @test get_covariance(mu_static) ≈ get_covariance(mu)
    @test allocations_measurement_update(
        x,
        P_static,
        y,
        h,
        as_cov(r),
        WanMerweWeightingParameters(),
    ) == 0
end

@testset "Static UKF with a model that returns a regular vector" begin
    x = @SVector randn(3)
    A = @SMatrix randn(3, 3)
    P = A'A + 3I
    F = randn(3, 3)
    f(x) = F * Vector(x)

    tu = time_update(Vector(x), Matrix(P), F, Matrix(P))
    tu_static = time_update(x, P, f, P)
    @test get_state(tu_static) ≈ get_state(tu)
    @test get_covariance(tu_static) ≈ get_covariance(tu)
end
