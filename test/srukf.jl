@testset "Square root Unscented Kalman filter" begin
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

    @testset "Rank-k Cholesky downdate with $T, uplo $uplo" for T in (Float64, ComplexF64),
        uplo in (:U, :L)

        A = randn(T, 6, 6)
        P = Hermitian(A' * A + 6I)
        V = 0.3 .* randn(T, 6, 3)
        P_chol = cholesky(P)
        C = uplo === :U ? Cholesky(copy(P_chol.U), 'U', 0) : Cholesky(copy(P_chol.L), 'L', 0)
        expected = foldl(lowrankdowndate!, eachcol(copy(V)); init = copy(C))
        result = KalmanFilters.lowrankdowndate_columns!(copy(C), copy(V))
        @test result.factors ≈ expected.factors
        @test Matrix(result) ≈ P - V * V'
        # P - v * v' has the diagonal entry -P[1, 1] and can't be positive definite
        V_bad = zeros(T, 6, 1)
        V_bad[1] = sqrt(2 * real(P[1, 1]))
        @test_throws PosDefException KalmanFilters.lowrankdowndate_columns!(copy(C), V_bad)
    end

    @testset "Covariance" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        x = randn(5)
        PL_prior = randn(5, 5)
        P_prior = PL_prior * PL_prior'
        χ = KalmanFilters.calc_sigma_points(x, P_prior, weight_params)
        F = randn(5, 5)
        f(x) = F * x
        𝓨 = KalmanFilters.transform(f, χ)
        QL = randn(5, 5)
        Q = QL * QL'
        Q_chol = cholesky(Q)
        P = @inferred KalmanFilters.cov(𝓨, Q_chol)
        @test P.L * P.U ≈ KalmanFilters.cov(𝓨, Q)
    end

    @testset "Posterior covariance" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        x = randn(5)
        PL = randn(5, 5)
        P = PL * PL'
        P_chol = cholesky(P)
        χ = KalmanFilters.calc_sigma_points(x, P, weight_params)
        F = randn(3, 5)
        h(x) = F * x
        𝓨 = KalmanFilters.transform(h, χ)
        y_est = KalmanFilters.mean(𝓨)
        unbiased_𝓨 = KalmanFilters.substract_mean(𝓨, y_est)
        RL = randn(3, 3)
        R = RL * RL'
        R_chol = cholesky(R)
        S = KalmanFilters.cov(unbiased_𝓨, R)
        S_chol = KalmanFilters.cov(unbiased_𝓨, R_chol)
        Pᵪᵧ = KalmanFilters.cov(χ, unbiased_𝓨)
        K = Pᵪᵧ / S_chol
        K_temp, P_post = @inferred KalmanFilters.calc_kalman_gain_and_posterior_covariance(
            P_chol,
            Pᵪᵧ,
            S_chol,
            [],
        )
        @test K_temp ≈ K
        @test P_post.L * P_post.U ≈ KalmanFilters.calc_posterior_covariance(P, Pᵪᵧ, K, [])
    end

    @testset "Time update with $T type $t" for T in (Float64, ComplexF64),
        t in ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        A = t.mat(randn(T, 3, 3))
        P = A'A
        P_chol = cholesky(Hermitian(P))
        B = t.mat(randn(T, 3, 3))
        Q = B'B
        Q_chol = cholesky(Hermitian(Q))
        F = t.mat(randn(T, 3, 3))
        f(x) = F * x

        tu = @inferred time_update(x, P, F, Q)
        tu_chol = @inferred time_update(x, P_chol, f, Q_chol)
        @test @inferred(get_covariance(tu_chol)) ≈ get_covariance(tu)
        @test @inferred(get_state(tu_chol)) ≈ get_state(tu)

        if x isa Vector
            f!(y, x) = mul!(y, F, x)
            tu_inter = @inferred SRUKFTUIntermediate(T, 3)
            tu_chol_inplace = @inferred time_update!(tu_inter, x, P_chol, f!, Q_chol)
            @test @inferred(get_covariance(tu_chol_inplace)) ≈ get_covariance(tu)
            @test @inferred(get_state(tu_chol_inplace)) ≈ get_state(tu)
        end
    end

    @testset "Measurement update with $T type $t" for T in (Float64, ComplexF64),
        t in ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        A = t.mat(randn(T, 3, 3))
        P = A'A
        P_chol = cholesky(Hermitian(P))
        B = t.mat(randn(T, 3, 3))
        R = B'B
        y = t.vec(randn(T, 3))
        R_chol = cholesky(Hermitian(R))
        H = t.mat(randn(T, 3, 3))
        h(x) = H * x

        mu = @inferred measurement_update(x, P, y, H, R)
        mu_chol = @inferred measurement_update(x, P_chol, y, h, R_chol)
        @test @inferred(get_covariance(mu_chol)) ≈ get_covariance(mu)
        @test @inferred(get_state(mu_chol)) ≈ get_state(mu)

        if x isa Vector
            h!(y, x) = mul!(y, H, x)
            mu_inter = @inferred SRUKFMUIntermediate(T, 3, 3)
            mu_chol_inplace =
                @inferred measurement_update!(mu_inter, x, P_chol, y, h!, R_chol)
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
        h(x) = H * x

        mu = measurement_update(x, P, y, H, R)
        mu_chol = @inferred measurement_update(x, P_chol, y, h, R_chol)
        @test @inferred(get_covariance(mu_chol)) ≈ get_covariance(mu)
        @test @inferred(get_state(mu_chol)) ≈ get_state(mu)
    end

    # Large enough for the blocked QR kernels (`tpqrt`/`geqrt`) to be used
    @testset "Updates with $num_x states and $num_y measurements with $T" for T in (
            Float64,
            ComplexF64,
        ),
        (num_x, num_y) in ((40, 36), (70, 8))

        random_pos_def(n) = (A = randn(T, n, n); Hermitian(A'A + n * I))
        x = randn(T, num_x)
        P = random_pos_def(num_x)
        Q = random_pos_def(num_x)
        R = random_pos_def(num_y)
        y = randn(T, num_y)
        F = randn(T, num_x, num_x)
        H = randn(T, num_y, num_x)
        f(x) = F * x
        f(x, noise) = F * x .+ noise
        f!(y, x) = mul!(y, F, x)
        f!(y, x, noise) = (mul!(y, F, x); y .+= noise)
        h(x) = H * x
        h(x, noise) = H * x .+ noise
        h!(y, x) = mul!(y, H, x)
        h!(y, x, noise) = (mul!(y, H, x); y .+= noise)

        tu = time_update(x, Matrix(P), F, Matrix(Q))
        mu = measurement_update(x, Matrix(P), y, H, Matrix(R))
        for (TU, MU, noise) in (
            (SRUKFTUIntermediate, SRUKFMUIntermediate, identity),
            (SRAUKFTUIntermediate, SRAUKFMUIntermediate, Augment),
        )
            tu_alloc = time_update(x, cholesky(P), f, noise(cholesky(Q)))
            @test get_covariance(tu_alloc) ≈ get_covariance(tu)
            @test get_state(tu_alloc) ≈ get_state(tu)
            mu_alloc = measurement_update(x, cholesky(P), y, h, noise(cholesky(R)))
            @test get_covariance(mu_alloc) ≈ get_covariance(mu)
            @test get_state(mu_alloc) ≈ get_state(mu)
            tu_inplace = time_update!(TU(T, num_x), x, cholesky(P), f!, noise(cholesky(Q)))
            @test get_covariance(tu_inplace) ≈ get_covariance(tu)
            @test get_state(tu_inplace) ≈ get_state(tu)
            mu_inplace = measurement_update!(
                MU(T, num_x, num_y),
                x,
                cholesky(P),
                y,
                h!,
                noise(cholesky(R)),
            )
            @test get_covariance(mu_inplace) ≈ get_covariance(mu)
            @test get_state(mu_inplace) ≈ get_state(mu)
        end
    end
end
