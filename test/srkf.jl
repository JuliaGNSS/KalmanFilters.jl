@testset "Square root Kalman filter" begin
    @testset "Calculate upper triangular of QR with $T" for T in (
        Float64,
        Float32,
        ComplexF64,
        ComplexF32,
    )
        A = randn(T, 10, 5)
        R = qr(A).R
        @test @inferred(KalmanFilters.calc_upper_triangular_of_qr(A)) ≈ R
        @test @inferred(KalmanFilters.calc_upper_triangular_of_qr!(copy(A))) ≈ R
        stacked = [A; triu(randn(T, 5, 5))]
        @test KalmanFilters.calc_upper_triangular_of_qr!(
            copy(stacked),
            KalmanFilters.calc_upper_triangular_of_stacked_qr_inplace!,
        ) ≈ qr(stacked).R
    end

    @testset "Calculate upper triangular of QR without LAPACK" begin
        A = randn(10, 5)
        R = qr(A).R
        @test KalmanFilters.calc_upper_triangular_of_qr(SMatrix{10,5}(A)) ≈ R
        @test KalmanFilters.calc_upper_triangular_of_qr!(big.(A)) ≈ R
    end

    # Sizes on both sides of the threshold at which the blocked LAPACK kernels take over
    @testset "Upper triangular of QR with $T, $num_dense_rows × $n dense rows" for T in (
            Float64,
            Float32,
            ComplexF64,
            ComplexF32,
        ),
        (num_dense_rows, n) in ((2, 1), (10, 5), (20, 10), (80, 40), (140, 70))

        U = triu(randn(T, n, n))
        A = [randn(T, num_dense_rows, n); U]
        A_dense = randn(T, num_dense_rows + n, n)
        workspace_length = @inferred KalmanFilters.calc_qr_workspace_length(A)
        tau = zeros(T, workspace_length)
        work = zeros(T, workspace_length)
        native_stacked!(R, A, tau, work) =
            KalmanFilters.householder_upper_triangular!(R, A, num_dense_rows)
        native_dense!(R, A, tau, work) =
            KalmanFilters.householder_upper_triangular!(R, A, size(A, 1))
        for (qr!, B) in (
            (KalmanFilters.calc_upper_triangular_of_stacked_qr_inplace!, A),
            (native_stacked!, A),
            (KalmanFilters.mytpqrt!, A),
            (KalmanFilters.calc_upper_triangular_of_dense_qr_inplace!, A_dense),
            (native_dense!, A_dense),
            (KalmanFilters.mygeqrt!, A_dense),
        )
            R = zeros(T, n, n)
            @test qr!(R, copy(B), tau, work) === R
            @test istriu(R)
            @test abs.(R) ≈ abs.(qr(B).R)
            @test R' * R ≈ B' * B
        end
        @test_throws DimensionMismatch KalmanFilters.mytpqrt!(
            zeros(T, n, n),
            copy(A),
            zeros(T, n - 1),
            work,
        )
    end

    @testset "Native upper triangular of QR with entries of magnitude $scale" for scale in (
        1e-170,
        1e170,
    )
        A = scale .* [randn(6, 3); triu(randn(3, 3))]
        R = KalmanFilters.householder_upper_triangular!(zeros(3, 3), copy(A), 6)
        @test all(isfinite, R)
        @test abs.(R) ≈ abs.(qr(A).R)
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

            # A reused buffer holds the previous call's QR factors, and a fresh one
            # whatever memory it was handed; neither may leak into the update.
            fill!(mu_interm.m, 7)
            mu_chol_reused = measurement_update!(mu_interm, x, P_chol, y, H, R_chol)
            @test get_covariance(mu_chol_reused) ≈ get_covariance(mu)
            @test get_state(mu_chol_reused) ≈ get_state(mu)
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

    # Covers the native QR and the blocked `tpqrt` (time update)/`geqrt` (measurement update)
    @testset "Updates with $num_x states and $num_y measurements with $T" for T in (
            Float64,
            ComplexF64,
        ),
        (num_x, num_y) in ((10, 4), (70, 8), (100, 30))

        random_pos_def(n) = (A = randn(T, n, n); Hermitian(A'A + n * I))
        x = randn(T, num_x)
        P = random_pos_def(num_x)
        Q = random_pos_def(num_x)
        R = random_pos_def(num_y)
        y = randn(T, num_y)
        F = randn(T, num_x, num_x)
        H = randn(T, num_y, num_x)

        tu = time_update(x, Matrix(P), F, Matrix(Q))
        tu_alloc = time_update(x, cholesky(P), F, cholesky(Q))
        tu_inplace =
            time_update!(SRKFTUIntermediate(T, num_x), x, cholesky(P), F, cholesky(Q))
        for tu_chol in (tu_alloc, tu_inplace)
            @test get_covariance(tu_chol) ≈ get_covariance(tu)
            @test get_state(tu_chol) ≈ get_state(tu)
        end

        mu = measurement_update(x, Matrix(P), y, H, Matrix(R))
        mu_alloc = measurement_update(x, cholesky(P), y, H, cholesky(R))
        mu_inplace = measurement_update!(
            SRKFMUIntermediate(T, num_x, num_y),
            x,
            cholesky(P),
            y,
            H,
            cholesky(R),
        )
        for mu_chol in (mu_alloc, mu_inplace)
            @test get_covariance(mu_chol) ≈ get_covariance(mu)
            @test get_state(mu_chol) ≈ get_state(mu)
        end
    end
end
