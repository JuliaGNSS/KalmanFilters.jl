# The static UKF, SR-UKF and QR don't allocate. The square root filter and the QR run on
# mutable copies, which stay on the stack only if they don't escape. With forced bounds
# checks (`--check-bounds=yes`) every index into them can throw a `BoundsError` holding the
# copy, so it escapes; `runtests.jl` then runs this file in a process of its own.
using Test, KalmanFilters, LinearAlgebra, StaticArrays

function allocations_time_update(x, P, f::F, Q, weight_params) where {F}
    time_update(x, P, f, Q, weight_params)
    return @allocated time_update(x, P, f, Q, weight_params)
end

function allocations_measurement_update(x, P, y, h::F, R, weight_params) where {F}
    measurement_update(x, P, y, h, R; weight_params)
    return @allocated measurement_update(x, P, y, h, R; weight_params)
end

function allocations_qr(A, num_dense_rows)
    KalmanFilters.calc_upper_triangular_of_qr(A, num_dense_rows)
    return @allocated KalmanFilters.calc_upper_triangular_of_qr(A, num_dense_rows)
end

@testset "Static $name with $T, $num_x states and $num_y measurements doesn't allocate" for (
        name,
        as_cov,
    ) in (
        ("UKF", identity),
        ("SRUKF", P -> cholesky(Hermitian(P))),
    ),
    T in (Float64, ComplexF64),
    (num_x, num_y) in ((1, 1), (2, 2), (4, 3), (12, 6))

    random_pos_def(n) = (A = @SMatrix(randn(T, n, n)); A'A + n * I)
    x = @SVector randn(T, num_x)
    P = as_cov(random_pos_def(num_x))
    Q = as_cov(random_pos_def(num_x))
    R = as_cov(random_pos_def(num_y))
    y = @SVector randn(T, num_y)
    F = @SMatrix randn(T, num_x, num_x)
    H = @SMatrix randn(T, num_y, num_x)
    f(x) = F * x
    h(x) = H * x
    # The weight of the mean sigma point is negative for the former and positive for the
    # latter, so the square root filter downdates and updates its factor respectively.
    for weight_params in (WanMerweWeightingParameters(), MeanSetWeightingParameters())
        @test allocations_time_update(x, P, f, Q, weight_params) == 0
        @test allocations_measurement_update(x, P, y, h, R, weight_params) == 0
    end
end

@testset "Static scalar $name measurement update doesn't allocate" for (name, as_cov) in (
    ("UKF", identity),
    ("SRUKF", cholesky),
)
    x = @SVector randn(3)
    A = @SMatrix randn(3, 3)
    # `A'A` needn't come out exactly symmetric, which `cholesky` of an `SMatrix` checks.
    B = A'A
    P = as_cov((B + B') / 2 + 3I)
    H = (@SVector randn(3))'
    h(x) = H * x
    @test allocations_measurement_update(
        x,
        P,
        randn(),
        h,
        as_cov(2.0),
        WanMerweWeightingParameters(),
    ) == 0
end

@testset "Static QR with $T and $num_dense_rows × $n dense rows doesn't allocate" for T in (
        Float64,
        ComplexF64,
    ),
    (num_dense_rows, n) in ((2, 1), (4, 2), (24, 12))

    A = SMatrix{num_dense_rows + n,n}([randn(T, num_dense_rows, n); triu(randn(T, n, n))])
    @test allocations_qr(A, num_dense_rows) == 0
    @test allocations_qr(A, size(A, 1)) == 0
end
