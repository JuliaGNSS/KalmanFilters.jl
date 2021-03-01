
[![pipeline status](https://git.rwth-aachen.de/nav/KalmanFilter.jl/badges/master/pipeline.svg)](https://git.rwth-aachen.de/nav/KalmanFilter.jl/commits/master)
[![coverage report](https://git.rwth-aachen.de/nav/KalmanFilter.jl/badges/master/coverage.svg)](https://git.rwth-aachen.de/nav/KalmanFilter.jl/commits/master)
# KalmanFilter
Provides multiple Kalman Filters

## Features
* (Square Root) Kalman Filter ((SR-)KF)
* (Square Root) Unscented Kalman Filter ((SR-)UKF)
* (Square Root) Augment Unscented Kalman Filter ((SR-)AUKF)

## Getting started

Install:
```julia
julia> ]
pkg> add git@git.rwth-aachen.de:nav/KalmanFilter.jl.git
```

## Usage

This package makes usage of multiple dispatch in Julia. 

If you'd like to use the (linear) Kalman-Filter, simply pass matrices for time update and measurement update function. If you'd like to use the Unscented-Kalman-Filter, pass functions to the time update and measurement update instead. You can also use the (linear) Kalman-Filter for the time update and the Unscented-Kalman-Filter for the measurement update or vice-versa.

If you pass cholesky decompositions for the state-covariance and the noise-covariante, this implementation will use the corresponding square-root variant.

If you like, you can augment the noise-covariance to the state-covariance, if you wrap the noise-covariance by `Augment`.

### Linear case
The linear Kalman Filter will be applied if you pass matrices `F` and `H` to the functions `time_update` and `measurement_update` respectively.
```julia
using KalmanFilter
Δt = 0.1
F = [1 Δt Δt^2/2; 0 1 Δt; 0 0 1]
H = [1, 0, 0]'
Q = [Δt^2/2; Δt; 1] * [Δt^2/2 Δt 1] * σ_acc_noise^2
R = σ_meas_noise^2
x_init = [0.0, 0.0, 0.0]
P_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
# Take first measurement
mu = measurement_update(x_init, P_init, measurement, H, R)
for i = 1:100
    # Take a measurement
    tu = time_update(get_state(mu), get_covariance(mu), F, Q)
    mu = measurement_update(get_state(tu), get_covariance(tu), measurement, H, R)
end
```
### Non-linear case
If you pass a function instead of matrix for `F` or `H`, the Unscented-Kalman-Filter will be used.
```julia
F(x) = x .* [1., 2.]
tu = time_update(x, P, F, Q)
```

#### Augmentation
KalmanFilter also allows to augment the noise-covariances:
```julia
F(x, noise) = x .* [1., 2.] .+ noise
tu = time_update(x, P, F, Augment(Q))
H(x, noise) = x .* [1., 1.] .+ noise
mu = measurement_update(x, P, measurement, H, Augment(R))
```

### Square Root Kalman filter
If you'd like to use the square root variant of the Kalman filter, you will have to pass the cholesky decomposition of the corresponding covariance, for e.g.:
```julia
using LinearAlgebra
P_init_chol = cholesky(P_init)
Q_chol = cholesky(Q)
R_chol = cholesky(R)
tu = time_update(x_init, P_init_chol, F, Q_chol)
mu = measurement_update(get_state(tu), get_sqrt_covariance(tu), measurement, H, Augment(R_chol))
```

### Statistical consistency testing
Goal: perform two of the most relevant consistency statistical tests of the Kalman filter
- the Normalized innovation squared (NIS) test
  - tests for unbiasedness
- the Innovation magnitude bound test
  - tests if approximately 68% (95%) of the innovation sequence values lie within the ⨦σ (⨦2σ) bound

## License

MIT License
