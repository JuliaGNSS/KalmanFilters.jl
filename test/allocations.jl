# The in-place updates must not allocate. Models are closures, as users typically write
# them: closures are `Function`s, which Julia only specialises on when the signature
# asks for it (`f!::F ... where {F}`), and without that every call boxes its arguments.
function make_linear_model(A)
    model!(y, x) = mul!(y, A, x)
    model!(y, x, noise) = (mul!(y, A, x); y .+= noise)
    return model!
end

# The updates write into `x` and `P`, so each call gets its own copy of the prior.
function allocations_time_update!(tu, x, P, f!::F, Q) where {F}
    time_update!(tu, copy(x), copy(P), f!, Q)
    x, P = copy(x), copy(P)
    return @allocated time_update!(tu, x, P, f!, Q)
end

function allocations_measurement_update!(mu, x, P, y, h!::F, R) where {F}
    measurement_update!(mu, copy(x), copy(P), y, h!, R)
    x, P = copy(x), copy(P)
    return @allocated measurement_update!(mu, x, P, y, h!, R)
end

@testset "In-place updates do not allocate ($num_x states, $num_y measurements)" for (
    num_x,
    num_y,
) in ((2, 2), (10, 4), (40, 36), (70, 8), (100, 30))
    random_pos_def(n) = (A = randn(n, n); A'A + n * I)
    x = randn(num_x)
    P = random_pos_def(num_x)
    Q = random_pos_def(num_x)
    R = random_pos_def(num_y)
    y = randn(num_y)
    F = randn(num_x, num_x)
    H = randn(num_y, num_x)
    f! = make_linear_model(F)
    h! = make_linear_model(H)
    P_chol = cholesky(P)
    Q_chol = cholesky(Q)
    R_chol = cholesky(R)
    # ForwardDiff only computes the Jacobian without allocating in vector mode, i.e. with
    # the chunk size being the number of states, which the default picks up to 12 states.
    backend = AutoForwardDiff(; chunksize = num_x)
    f_jacobian = JacobianPreparation(f!, zeros(num_x), zeros(num_x); backend)
    h_jacobian = JacobianPreparation(h!, zeros(num_y), zeros(num_x); backend)

    @test allocations_time_update!(KFTUIntermediate(num_x), x, P, F, Q) == 0
    @test allocations_time_update!(EKFTUIntermediate(num_x), x, P, f_jacobian, Q) == 0
    @test allocations_time_update!(SRKFTUIntermediate(num_x), x, P_chol, F, Q_chol) == 0
    @test allocations_time_update!(
        SREKFTUIntermediate(num_x),
        x,
        P_chol,
        f_jacobian,
        Q_chol,
    ) == 0
    @test allocations_time_update!(UKFTUIntermediate(num_x), x, P, f!, Q) == 0
    @test allocations_time_update!(SRUKFTUIntermediate(num_x), x, P_chol, f!, Q_chol) == 0
    @test allocations_time_update!(AUKFTUIntermediate(num_x), x, P, f!, Augment(Q)) == 0
    @test allocations_time_update!(
        SRAUKFTUIntermediate(num_x),
        x,
        P_chol,
        f!,
        Augment(Q_chol),
    ) == 0

    @test allocations_measurement_update!(KFMUIntermediate(num_x, num_y), x, P, y, H, R) ==
          0
    @test allocations_measurement_update!(
        EKFMUIntermediate(num_x, num_y),
        x,
        P,
        y,
        h_jacobian,
        R,
    ) == 0
    @test allocations_measurement_update!(
        SRKFMUIntermediate(num_x, num_y),
        x,
        P_chol,
        y,
        H,
        R_chol,
    ) == 0
    @test allocations_measurement_update!(
        SREKFMUIntermediate(num_x, num_y),
        x,
        P_chol,
        y,
        h_jacobian,
        R_chol,
    ) == 0
    @test allocations_measurement_update!(UKFMUIntermediate(num_x, num_y), x, P, y, h!, R) ==
          0
    @test allocations_measurement_update!(
        SRUKFMUIntermediate(num_x, num_y),
        x,
        P_chol,
        y,
        h!,
        R_chol,
    ) == 0
    @test allocations_measurement_update!(
        AUKFMUIntermediate(num_x, num_y),
        x,
        P,
        y,
        h!,
        Augment(R),
    ) == 0
    @test allocations_measurement_update!(
        SRAUKFMUIntermediate(num_x, num_y),
        x,
        P_chol,
        y,
        h!,
        Augment(R_chol),
    ) == 0
end

@testset "In-place square-root sigma point updates with lower Cholesky factors" begin
    random_pos_def(n) = (A = randn(n, n); A'A + n * I)
    num_x, num_y = 10, 4
    x = randn(num_x)
    P = random_pos_def(num_x)
    Q = random_pos_def(num_x)
    R = random_pos_def(num_y)
    y = randn(num_y)
    f! = make_linear_model(randn(num_x, num_x))
    h! = make_linear_model(randn(num_y, num_x))
    upper(A) = cholesky(Hermitian(A, :U))
    lower(A) = cholesky(Hermitian(A, :L))

    @testset "$name" for (name, TU, MU, noise) in (
        ("SRUKF", SRUKFTUIntermediate, SRUKFMUIntermediate, identity),
        ("SRAUKF", SRAUKFTUIntermediate, SRAUKFMUIntermediate, Augment),
    )
        tu_upper = time_update!(TU(num_x), copy(x), upper(P), f!, noise(upper(Q)))
        tu_lower = time_update!(TU(num_x), copy(x), lower(P), f!, noise(lower(Q)))
        @test get_state(tu_lower) ≈ get_state(tu_upper)
        @test get_covariance(tu_lower) ≈ get_covariance(tu_upper)
        @test allocations_time_update!(TU(num_x), x, lower(P), f!, noise(lower(Q))) == 0

        mu_upper = measurement_update!(
            MU(num_x, num_y),
            copy(x),
            upper(P),
            y,
            h!,
            noise(upper(R)),
        )
        mu_lower = measurement_update!(
            MU(num_x, num_y),
            copy(x),
            lower(P),
            y,
            h!,
            noise(lower(R)),
        )
        @test get_state(mu_lower) ≈ get_state(mu_upper)
        @test get_covariance(mu_lower) ≈ get_covariance(mu_upper)
        @test allocations_measurement_update!(
            MU(num_x, num_y),
            x,
            lower(P),
            y,
            h!,
            noise(lower(R)),
        ) == 0
    end
end

@testset "SRUKF in-place gain with a lower innovation factor" begin
    # The filters build the innovation factor as an upper Cholesky, so exercise the lower
    # branch of the gain computation directly.
    random_pos_def(n) = (A = randn(n, n); A'A + n * I)
    num_x, num_y = 10, 4
    P = cholesky(random_pos_def(num_x))
    S = random_pos_def(num_y)
    Pᵪᵧ = randn(num_x, num_y)
    gain!(S_chol) = KalmanFilters.calc_kalman_gain_and_posterior_covariance!(
        zeros(num_x, num_y),
        zeros(num_x),
        copy(P),
        copy(Pᵪᵧ),
        S_chol,
    )
    K_upper, P_upper = gain!(cholesky(Hermitian(S, :U)))
    K_lower, P_lower = gain!(cholesky(Hermitian(S, :L)))
    @test K_lower ≈ K_upper
    @test Matrix(P_lower) ≈ Matrix(P_upper)
end
