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

    @testset "In-place updates with $T" for T = (Float64,)
        num_x, num_y = 4, 3
        x = randn(T, 4)
        PL = randn(T, 4, 4)
        P = PL'PL + I
        QL = randn(T, 4, 4)
        Q = QL'QL
        RL = randn(T, 3, 3)
        R = RL'RL + I
        y = randn(T, 3)
        F = randn(T, 4, 4)
        H = randn(T, 3, 4)
        a = randn(T, 4)
        b = randn(T, 3)
        f(x, a) = F * x + a
        f!(out, x, a) = (mul!(out, F, x); out .+= a)
        h(x, b) = H * x + b
        h!(out, x, b) = (mul!(out, H, x); out .+= b)

        f_jac = JacobianPreparation(f, zero(x), Constant(a))
        h_jac = JacobianPreparation(h, zero(x), Constant(b))
        f!_jac = JacobianPreparation(f!, zeros(T, num_x), zero(x), Constant(a))
        h!_jac = JacobianPreparation(h!, zeros(T, num_y), zero(x), Constant(b))

        x_inplace, P_inplace = copy(x), copy(P)
        tu = time_update(x, P, f_jac, Q)
        tu! = time_update!(EKFTUIntermediate(T, num_x), x_inplace, P_inplace, f!_jac, Q)
        @test get_state(tu!) === x_inplace
        @test get_covariance(tu!) === P_inplace
        @test x_inplace ≈ get_state(tu)
        @test P_inplace ≈ get_covariance(tu)

        mu = measurement_update(get_state(tu), get_covariance(tu), y, h_jac, R)
        mu! = measurement_update!(
            EKFMUIntermediate(T, num_x, num_y),
            x_inplace,
            P_inplace,
            y,
            h!_jac,
            R,
        )
        @test get_state(mu!) === x_inplace
        @test get_covariance(mu!) === P_inplace
        @test x_inplace ≈ get_state(mu)
        @test P_inplace ≈ get_covariance(mu)
        @test get_innovation(mu!) ≈ get_innovation(mu)
        @test get_innovation_covariance(mu!) ≈ get_innovation_covariance(mu)
        @test get_kalman_gain(mu!) ≈ get_kalman_gain(mu)

        # The in-place preparation takes changed contexts like the allocating one.
        a2 = randn(T, 4)
        f_jac2 = GradientOrJacobianContextUpdate(f_jac, Constant(a2))
        f!_jac2 = GradientOrJacobianContextUpdate(f!_jac, Constant(a2))
        tu2 = time_update(get_state(mu), get_covariance(mu), f_jac2, Q)
        time_update!(EKFTUIntermediate(T, num_x), x_inplace, P_inplace, f!_jac2, Q)
        @test x_inplace ≈ get_state(tu2)
        @test P_inplace ≈ get_covariance(tu2)
    end
end
