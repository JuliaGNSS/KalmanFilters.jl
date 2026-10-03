# The in-place updates write the new state into `x` and the new covariance into `P`,
# a `Cholesky` into whichever factor it stores, and return an update whose state and
# covariance are those very objects, so a filter loop reads nothing back.
@testset "In-place updates write into x and P ($name)" for (name, TU, MU, chol, noise) in (
    ("KF", KFTUIntermediate, KFMUIntermediate, false, identity),
    ("EKF", EKFTUIntermediate, EKFMUIntermediate, false, identity),
    ("SRKF", SRKFTUIntermediate, SRKFMUIntermediate, true, identity),
    ("SREKF", SREKFTUIntermediate, SREKFMUIntermediate, true, identity),
    ("UKF", UKFTUIntermediate, UKFMUIntermediate, false, identity),
    ("SRUKF", SRUKFTUIntermediate, SRUKFMUIntermediate, true, identity),
    ("AUKF", AUKFTUIntermediate, AUKFMUIntermediate, false, Augment),
    ("SRAUKF", SRAUKFTUIntermediate, SRAUKFMUIntermediate, true, Augment),
)
    random_pos_def(n) = (A = randn(n, n); A'A + n * I)
    num_x, num_y = 5, 3
    x0 = randn(num_x)
    P0 = random_pos_def(num_x)
    Q = random_pos_def(num_x)
    R = random_pos_def(num_y)
    y = randn(num_y)
    F = randn(num_x, num_x)
    H = randn(num_y, num_x)
    f! = make_linear_model(F)
    h! = make_linear_model(H)
    # The linear filters take the matrices, the EKF the Jacobians of the models and the
    # sigma point filters the models.
    f_arg, h_arg = if name in ("KF", "SRKF")
        F, H
    elseif name in ("EKF", "SREKF")
        JacobianPreparation(f!, zeros(num_x), zeros(num_x)),
        JacobianPreparation(h!, zeros(num_y), zeros(num_x))
    else
        f!, h!
    end

    expected_tu = time_update(x0, P0, F, Q)
    expected_mu =
        measurement_update(get_state(expected_tu), get_covariance(expected_tu), y, H, R)

    @testset "$(uplo) factor" for uplo in (chol ? (:U, :L) : (:none,))
        as_cov(A) = uplo === :none ? Matrix(A) : cholesky(Hermitian(A, uplo))
        x = copy(x0)
        P = as_cov(P0)
        tu = time_update!(TU(num_x), x, P, f_arg, noise(as_cov(Q)))
        @test get_state(tu) === x
        @test tu.covariance === P
        @test x ≈ get_state(expected_tu)
        @test Matrix(P) ≈ get_covariance(expected_tu)
        if chol
            @test P.uplo === (uplo === :U ? 'U' : 'L')
        end

        mu = measurement_update!(MU(num_x, num_y), x, P, y, h_arg, noise(as_cov(R)))
        @test get_state(mu) === x
        @test mu.covariance === P
        @test x ≈ get_state(expected_mu)
        @test Matrix(P) ≈ get_covariance(expected_mu)
    end
end
