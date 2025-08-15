@testset "Extended Kalman filter" begin

    @testset "Time update with $T type $t" for T = (Float64, ComplexF64), t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        QL = t.mat(randn(T, 3, 3))
        Q = QL'QL
        F = t.mat(randn(T, 3, 3))
        f(x) = F * x

        tu = time_update(x, P, F, Q)
        x_apri = f(x)
        tu_ekf = time_update(x, x_apri, P, F, Q)
        @test get_covariance(tu_ekf) ≈ get_covariance(tu)
        @test get_state(tu_ekf) ≈ get_state(tu)

    end

    @testset "Measurement update with $T type $t" for T = (Float64, ComplexF64), t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        RL = t.mat(randn(T, 3, 3))
        R = RL'RL
        y = t.vec(randn(T, 3))
        H = t.mat(randn(T, 3, 3))
        h(x) = H * x

        mu = measurement_update(x, P, y, H, R)
        y_pre = h(x)
        mu_ekf = @inferred measurement_update(x, P, y, y_pre, H, R)
        @test get_covariance(mu_ekf) ≈ get_covariance(mu)
        @test get_state(mu_ekf) ≈ get_state(mu)
        
    end

    @testset "Scalar measurement update with $T type $t" for T = (Float64, ComplexF64), t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))
        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        RL = randn()
        R = RL'RL
        y = randn(T)
        H = t.vec(randn(T, 3))'
        h(x) = H * x

        mu = measurement_update(x, P, y, H, R)
        y_pre = h(x)
        mu_ekf = @inferred measurement_update(x, P, y, y_pre, H, R)
        @test @inferred(get_covariance(mu_ekf)) ≈ get_covariance(mu)
        @test @inferred(get_state(mu_ekf)) ≈ get_state(mu)
    end
end
