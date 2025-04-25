using KalmanFilters
using LinearAlgebra
using StaticArrays

Dx = 2
Dy = 2

F = @SMatrix rand(Dx, Dx)
Q = @SMatrix rand(Dx, Dx)
Q = Q * Q'
H = @SMatrix rand(Dy, Dx)
R = @SMatrix rand(Dy, Dy)
R = R * R'
x_init = @SVector rand(Dx)
P_init = @SMatrix rand(Dx, Dx)
P_init = P_init * P_init'
measurement = @SVector rand(Dy)

P_init_chol = cholesky(P_init)
Q_chol = cholesky(Q)
R_chol = cholesky(R)
tu = time_update(x_init, P_init_chol, F, Q_chol)
mu = measurement_update(get_state(tu), get_sqrt_covariance(tu), measurement, H, R_chol)

# Confirm output type is correct
println("Output Type: ", typeof(mu.covariance))

# Benchmark against regular array version
using BenchmarkTools

println("StaticArrays:")
res_static = @benchmark begin
    tu = time_update($x_init, $P_init_chol, $F, $Q_chol)
    mu = measurement_update(get_state(tu), get_sqrt_covariance(tu), $measurement, $H, $R_chol)
    for i in 1:100
        tu = time_update(get_state(tu), get_sqrt_covariance(tu), $F, $Q_chol)
        mu = measurement_update(get_state(tu), get_sqrt_covariance(tu), $measurement, $H, $R_chol)
    end
end
display(res_static)

println("Regular Arrays:")
F_reg = rand(Dx, Dx)
Q_reg = rand(Dx, Dx)
Q_reg = Q_reg * Q_reg'
H_reg = rand(Dy, Dx)
R_reg = rand(Dy, Dy)
R_reg = R_reg * R_reg'
x_init_reg = rand(Dx)
P_init_reg = rand(Dx, Dx)
P_init_reg = P_init_reg * P_init_reg'
measurement_reg = rand(Dy)

P_init_chol_reg = cholesky(P_init_reg)
Q_chol_reg = cholesky(Q_reg)
R_chol_reg = cholesky(R_reg)

res_regular = @benchmark begin
    tu = time_update($x_init_reg, $P_init_chol_reg, $F_reg, $Q_chol_reg)
    mu = measurement_update(get_state(tu), get_sqrt_covariance(tu), $measurement_reg, $H_reg, $R_chol_reg)
    for i in 1:100
        tu = time_update(get_state(tu), get_sqrt_covariance(tu), $F_reg, $Q_chol_reg)
        mu = measurement_update(get_state(tu), get_sqrt_covariance(tu), $measurement_reg, $H_reg, $R_chol_reg)
    end
end
display(res_regular)

# Print speed-up
println("Speed-up: ", mean(res_regular.times) / mean(res_static.times[1]))
