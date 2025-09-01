using ForwardDiff
using DifferentiationInterface
@testset "Extended Kalman filter" begin

    @testset "Time update with $T type $t" for T = (Float64,), t = ((vec=Vector, mat=Matrix), (vec=SVector{3}, mat=SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        QL = t.mat(randn(T, 3, 3))
        Q = QL'QL
        F = t.mat(randn(T, 3, 3))
        f(x) = F * x

        jacobian_preparation = JacobianPreparation(f, zero(x))

        tu = time_update(x, P, F, Q)
        tu_ekf = time_update(x, P, jacobian_preparation, Q)
        @test get_covariance(tu_ekf) ≈ get_covariance(tu)
        @test get_state(tu_ekf) ≈ get_state(tu)
    end

    @testset "Time update with constant context and with $T type $t" for T = (Float64,), t = ((vec=Vector, mat=Matrix), (vec=SVector{3}, mat=SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        QL = t.mat(randn(T, 3, 3))
        Q = QL'QL
        F = t.mat(randn(T, 3, 3))
        a = zeros(T, 3)
        f(x, a) = F * x + a

        jacobian_preparation = JacobianPreparation(f, zero(x), Constant(a))

        tu = time_update(x, P, F, Q)
        tu_ekf = time_update(x, P, jacobian_preparation, Q)

        @test get_covariance(tu_ekf) ≈ get_covariance(tu)
        @test get_state(tu_ekf) ≈ get_state(tu)
    end

    @testset "Time update with multiple constant contexts and with $T type $t" for T = (Float64,), t = ((vec=Vector, mat=Matrix), (vec=SVector{3}, mat=SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        QL = t.mat(randn(T, 3, 3))
        Q = QL'QL
        F = t.mat(randn(T, 3, 3))
        a = zeros(T, 3)
        b = zeros(T, 3)
        f(x, a, b) = F * x + a + b

        jacobian_preparation = JacobianPreparation(f, zero(x), Constant(a), Constant(b))

        tu = time_update(x, P, F, Q)
        tu_ekf = time_update(x, P, jacobian_preparation, Q)

        @test get_covariance(tu_ekf) ≈ get_covariance(tu)
        @test get_state(tu_ekf) ≈ get_state(tu)
    end

    @testset "Time update with changing context and with $T type $t" for T = (Float64,), t = ((vec=Vector, mat=Matrix), (vec=SVector{3}, mat=SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        QL = t.mat(randn(T, 3, 3))
        Q = QL'QL
        F = t.mat([1 0 0; 0 1 0; 0 0 1]);
        a = ones(T, 3)
        f(x, a) = F * x + a

        jacobian_preparation = JacobianPreparation(f, zero(x), Constant(a))

        tu = time_update(x, P, F, Q)
        tu_ekf = time_update(x, P, jacobian_preparation, Q)

        a *= -1
        jacobian_preparation = GradientOrJacobianContextUpdate(jacobian_preparation, Constant(a))

        tu = time_update(get_state(tu), get_covariance(tu), F, Q)
        tu_ekf = time_update(get_state(tu_ekf), get_covariance(tu_ekf), jacobian_preparation, Q)

        @test get_covariance(tu_ekf) ≈ get_covariance(tu)
        @test get_state(tu_ekf) ≈ get_state(tu)
    end

    @testset "Time update with multiple changing contexts and with $T type $t" for T = (Float64,), t = ((vec=Vector, mat=Matrix), (vec=SVector{3}, mat=SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        QL = t.mat(randn(T, 3, 3))
        Q = QL'QL
        F = t.mat([1 0 0; 0 1 0; 0 0 1]);
        a = ones(T, 3)
        b = ones(T, 3)
        f(x, a, b) = F * x + a + b

        jacobian_preparation = JacobianPreparation(f, zero(x), Constant(a), Constant(b))

        tu = time_update(x, P, F, Q)
        tu_ekf = time_update(x, P, jacobian_preparation, Q)

        a *= -1
        b *= -1
        jacobian_preparation = GradientOrJacobianContextUpdate(jacobian_preparation, Constant(a), Constant(b))

        tu = time_update(get_state(tu), get_covariance(tu), F, Q)
        tu_ekf = time_update(get_state(tu_ekf), get_covariance(tu_ekf), jacobian_preparation, Q)

        @test get_covariance(tu_ekf) ≈ get_covariance(tu)
        @test get_state(tu_ekf) ≈ get_state(tu)
    end

    @testset "Measurement update with $T type $t" for T = (Float64,), t = ((vec=Vector, mat=Matrix), (vec=SVector{3}, mat=SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        RL = t.mat(randn(T, 3, 3))
        R = RL'RL
        y = t.vec(randn(T, 3))
        H = t.mat(randn(T, 3, 3))
        h(x) = H * x

        jacobian_preparation = JacobianPreparation(h, zero(x))

        mu = measurement_update(x, P, y, H, R)
        mu_ekf = @inferred measurement_update(x, P, y, jacobian_preparation, R)
        @test get_covariance(mu_ekf) ≈ get_covariance(mu)
        @test get_state(mu_ekf) ≈ get_state(mu)

    end

    @testset "Measurement update with context and with $T type $t" for T = (Float64,), t = ((vec=Vector, mat=Matrix), (vec=SVector{3}, mat=SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        RL = t.mat(randn(T, 3, 3))
        R = RL'RL
        y = t.vec(randn(T, 3))
        H = t.mat(randn(T, 3, 3))
        a = zeros(T, 3)
        h(x, a) = H * x + a

        jacobian_preparation = JacobianPreparation(h, zero(x), Constant(a))

        mu = measurement_update(x, P, y, H, R)
        mu_ekf = @inferred measurement_update(x, P, y, jacobian_preparation, R)
        @test get_covariance(mu_ekf) ≈ get_covariance(mu)
        @test get_state(mu_ekf) ≈ get_state(mu)

    end

    @testset "Scalar measurement update with $T type $t" for T = (Float64,), t = ((vec=Vector, mat=Matrix), (vec=SVector{3}, mat=SMatrix{3,3}))
        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        RL = randn()
        R = RL'RL
        y = randn(T)
        H = t.vec(randn(T, 3))'
        h(x) = H * x

        gradient_preparation = GradientPreparation(h, zero(x))

        mu = measurement_update(x, P, y, H, R)
        mu_ekf = @inferred measurement_update(x, P, y, gradient_preparation, R)
        @test @inferred(get_covariance(mu_ekf)) ≈ get_covariance(mu)
        @test @inferred(get_state(mu_ekf)) ≈ get_state(mu)
    end
end
