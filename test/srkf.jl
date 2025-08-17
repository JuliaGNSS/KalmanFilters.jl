@testset "Square root Kalman filter" begin
    @testset "Calculate upper triangular of QR" begin
        A = randn(10, 5)
        R_test = @inferred KalmanFilters.calc_upper_triangular_of_qr!(copy(A))

        Q, R = qr(A)
        @test R_test ≈ R ≈ KalmanFilters.calc_upper_triangular_of_qr(A)

        qr_zeros = zeros(10)
        qr_space_length = @inferred KalmanFilters.calc_gels_working_size(A, qr_zeros)
        qr_space = zeros(qr_space_length)
        R_res = zeros(5, 5)
        R_test_inplace = @inferred KalmanFilters.calc_upper_triangular_of_qr_inplace!(
            R_res,
            copy(A),
            qr_zeros,
            qr_space,
        )
        @test R_test_inplace ≈ R
    end

    @testset "Time update with $T type $t" for T in (Float64, ComplexF64),
        t in ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        P_chol = cholesky(Hermitian(P))
        QL = t.mat(randn(T, 3, 3))
        Q = QL'QL
        Q_chol = cholesky(Hermitian(Q))
        F = t.mat(randn(T, 3, 3))

        tu = @inferred time_update(x, P, F, Q)
        tu_chol = @inferred time_update(x, P_chol, F, Q_chol)
        @test @inferred(get_covariance(tu_chol)) ≈ get_covariance(tu)
        @test @inferred(get_state(tu_chol)) ≈ get_state(tu)

        if x isa Vector
            tu_interm = @inferred SRKFTUIntermediate(T, 3)
            tu_chol_inplace = @inferred time_update!(tu_interm, x, P_chol, F, Q_chol)
            @test @inferred(get_covariance(tu_chol_inplace)) ≈ get_covariance(tu)
            @test @inferred(get_state(tu_chol_inplace)) ≈ get_state(tu)
        end
    end

    @testset "Measurement update with $T type $t" for T in (Float64, ComplexF64),
        t in ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        P_chol = cholesky(Hermitian(P))
        RL = t.mat(randn(T, 3, 3))
        R = RL'RL
        R_chol = cholesky(Hermitian(R))
        y = t.vec(randn(T, 3))
        H = t.mat(randn(T, 3, 3))

        mu = @inferred measurement_update(x, P, y, H, R)
        mu_chol = @inferred measurement_update(x, P_chol, y, H, R_chol)
        @test @inferred(get_covariance(mu_chol)) ≈ get_covariance(mu)
        @test @inferred(get_state(mu_chol)) ≈ get_state(mu)

        if x isa Vector
            mu_interm = SRKFMUIntermediate(T, 3, 3)
            mu_chol_inplace =
                @inferred measurement_update!(mu_interm, x, P_chol, y, H, R_chol)
            @test @inferred(get_covariance(mu_chol_inplace)) ≈ get_covariance(mu)
            @test @inferred(get_state(mu_chol_inplace)) ≈ get_state(mu)
        end
    end

    @testset "Scalar measurement update with $T type $t" for T in (Float64, ComplexF64),
        t in ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        P_chol = cholesky(Hermitian(P))
        RL = randn()
        R = RL'RL
        R_chol = cholesky(R)
        y = randn(T)
        H = t.vec(randn(T, 3))'

        mu = @inferred measurement_update(x, P, y, H, R)
        mu_chol = @inferred measurement_update(x, P_chol, y, H, R_chol)
        @test @inferred(get_covariance(mu_chol)) ≈ get_covariance(mu)
        @test @inferred(get_state(mu_chol)) ≈ get_state(mu)
    end
end
