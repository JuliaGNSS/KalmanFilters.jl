@testset "Square root extended Kalman filter" begin
    # With a linear model the SR-EKF must match the SR-KF on the model's matrix.
    @testset "Time update with $T type $t" for T = (Float64,),
        t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P_chol = cholesky(Hermitian(PL'PL + I))
        QL = t.mat(randn(T, 3, 3))
        Q_chol = cholesky(Hermitian(QL'QL + I))
        F = t.mat(randn(T, 3, 3))
        a = t.vec(randn(T, 3))
        f(x, a) = F * x + a

        jacobian_preparation = JacobianPreparation(f, zero(x), Constant(a))

        tu = time_update(x, P_chol, F, Q_chol)
        tu_ekf = @inferred time_update(x, P_chol, jacobian_preparation, Q_chol)
        @test tu_ekf.covariance isa Cholesky
        @test get_sqrt_covariance(tu_ekf).U ≈ get_sqrt_covariance(tu).U
        @test get_covariance(tu_ekf) ≈ get_covariance(tu)
        @test get_state(tu_ekf) ≈ get_state(tu) + a
    end

    @testset "Measurement update with $T type $t" for T = (Float64,),
        t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P_chol = cholesky(Hermitian(PL'PL + I))
        RL = t.mat(randn(T, 3, 3))
        R_chol = cholesky(Hermitian(RL'RL + I))
        y = t.vec(randn(T, 3))
        H = t.mat(randn(T, 3, 3))
        h(x) = H * x

        jacobian_preparation = JacobianPreparation(h, zero(x))

        mu = measurement_update(x, P_chol, y, H, R_chol)
        mu_ekf = @inferred measurement_update(x, P_chol, y, jacobian_preparation, R_chol)
        @test mu_ekf.covariance isa Cholesky
        @test get_covariance(mu_ekf) ≈ get_covariance(mu)
        @test get_state(mu_ekf) ≈ get_state(mu)
        @test get_innovation(mu_ekf) ≈ get_innovation(mu)
        @test get_innovation_covariance(mu_ekf) ≈ get_innovation_covariance(mu)
        @test get_kalman_gain(mu_ekf) ≈ get_kalman_gain(mu)
    end

    @testset "Scalar measurement update with $T type $t" for T = (Float64,),
        t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P_chol = cholesky(Hermitian(PL'PL + I))
        R_chol = cholesky(2.0)
        y = randn(T)
        H = t.vec(randn(T, 3))'
        h(x) = H * x

        gradient_preparation = GradientPreparation(h, zero(x))

        mu = measurement_update(x, P_chol, y, H, R_chol)
        mu_ekf = @inferred measurement_update(x, P_chol, y, gradient_preparation, R_chol)
        @test get_covariance(mu_ekf) ≈ get_covariance(mu)
        @test get_state(mu_ekf) ≈ get_state(mu)
    end

    @testset "Measurement update with considered states" begin
        x = randn(5)
        PL = randn(5, 5)
        P_chol = cholesky(Hermitian(PL'PL + I))
        RL = randn(2, 2)
        R_chol = cholesky(Hermitian(RL'RL + I))
        y = randn(2)
        H = randn(2, 5)
        h(x) = H * x

        mu = measurement_update(x, P_chol, y, H, R_chol; consider = 4:5)
        mu_ekf = measurement_update(
            x,
            P_chol,
            y,
            JacobianPreparation(h, zero(x)),
            R_chol;
            consider = 4:5,
        )
        @test get_covariance(mu_ekf) ≈ get_covariance(mu)
        @test get_state(mu_ekf) ≈ get_state(mu)
    end

    # A non-linear model: the SR-EKF must match the EKF on the same covariances.
    @testset "Non-linear updates match the EKF" begin
        x = randn(3)
        PL = randn(3, 3)
        P = PL'PL + I
        Q = 0.1 * I(3) + zeros(3, 3)
        R = [0.5 0.1; 0.1 0.4]
        y = randn(2)
        f(x) = [x[1] + 0.1 * sin(x[2]), x[2] + 0.1 * x[3]^2, 0.9 * x[3]]
        h(x) = [x[1]^2 + x[2], cos(x[3])]
        f_jac = JacobianPreparation(f, zero(x))
        h_jac = JacobianPreparation(h, zero(x))

        tu = time_update(x, P, f_jac, Q)
        tu_sr = time_update(x, cholesky(Hermitian(P)), f_jac, cholesky(Hermitian(Q)))
        @test get_state(tu_sr) ≈ get_state(tu)
        @test get_covariance(tu_sr) ≈ get_covariance(tu)

        mu = measurement_update(get_state(tu), get_covariance(tu), y, h_jac, R)
        mu_sr = measurement_update(
            get_state(tu_sr),
            get_sqrt_covariance(tu_sr),
            y,
            h_jac,
            cholesky(Hermitian(R)),
        )
        @test get_state(mu_sr) ≈ get_state(mu)
        @test get_covariance(mu_sr) ≈ get_covariance(mu)
        @test get_kalman_gain(mu_sr) ≈ get_kalman_gain(mu)
    end

    @testset "In-place updates with $(uplo) factors" for uplo in (:U, :L)
        num_x, num_y = 4, 3
        x = randn(num_x)
        PL = randn(num_x, num_x)
        P = PL'PL + I
        QL = randn(num_x, num_x)
        Q = QL'QL
        RL = randn(num_y, num_y)
        R = RL'RL + I
        y = randn(num_y)
        a = randn(num_x)
        b = randn(num_y)
        f(x, a) = [x[1] + 0.1 * sin(x[2]), x[2] + 0.1 * x[3]^2, 0.9 * x[3], x[4]] + a
        f!(out, x, a) = (out .= f(x, a))
        h(x, b) = [x[1]^2 + x[2], cos(x[3]), x[4]] + b
        h!(out, x, b) = (out .= h(x, b))
        as_chol(A) = cholesky(Hermitian(A, uplo))

        f_jac = JacobianPreparation(f, zero(x), Constant(a))
        h_jac = JacobianPreparation(h, zero(x), Constant(b))
        f!_jac = JacobianPreparation(f!, zeros(num_x), zero(x), Constant(a))
        h!_jac = JacobianPreparation(h!, zeros(num_y), zero(x), Constant(b))

        x_inplace, P_inplace = copy(x), as_chol(P)
        tu = time_update(x, as_chol(P), f_jac, as_chol(Q))
        tu! = @inferred time_update!(
            SREKFTUIntermediate(num_x),
            x_inplace,
            P_inplace,
            f!_jac,
            as_chol(Q),
        )
        @test get_state(tu!) === x_inplace
        @test tu!.covariance === P_inplace
        @test P_inplace.uplo === (uplo === :U ? 'U' : 'L')
        @test x_inplace ≈ get_state(tu)
        @test get_covariance(tu!) ≈ get_covariance(tu)

        mu = measurement_update(get_state(tu), tu.covariance, y, h_jac, as_chol(R))
        mu_interm = SREKFMUIntermediate(num_x, num_y)
        mu! = @inferred measurement_update!(
            mu_interm,
            x_inplace,
            P_inplace,
            y,
            h!_jac,
            as_chol(R),
        )
        @test get_state(mu!) === x_inplace
        @test mu!.covariance === P_inplace
        @test x_inplace ≈ get_state(mu)
        @test get_covariance(mu!) ≈ get_covariance(mu)
        @test get_innovation(mu!) ≈ get_innovation(mu)
        @test get_innovation_covariance(mu!) ≈ get_innovation_covariance(mu)
        @test get_kalman_gain(mu!) ≈ get_kalman_gain(mu)

        # A reused buffer holds the previous call's QR factors; they may not leak into
        # the update.
        fill!(mu_interm.srkf.m, 7)
        x_reused, P_reused = copy(get_state(tu)), copy(tu.covariance)
        measurement_update!(mu_interm, x_reused, P_reused, y, h!_jac, as_chol(R))
        @test x_reused ≈ get_state(mu)
        @test Matrix(P_reused) ≈ get_covariance(mu)
    end
end
