
[![pipeline status](https://git.rwth-aachen.de/nav/KalmanFilter.jl/badges/master/pipeline.svg)](https://git.rwth-aachen.de/nav/KalmanFilter.jl/commits/master)
[![coverage report](https://git.rwth-aachen.de/nav/KalmanFilter.jl/badges/master/coverage.svg)](https://git.rwth-aachen.de/nav/KalmanFilter.jl/commits/master)
# KalmanFilter
Provides multiple Kalman Filters

## Features
* Kalman Filter (KF)
* Unscented Kalman Filter (UKF)
* Augment Unscented Kalman Filter (AUKF)

## Getting started

Install:
```julia
julia> ]
pkg> add git@git.rwth-aachen.de:nav/KalmanFilter.jl.git
```

## Usage

### Linear case
The linear Kalman Filter will be applied if you pass matrices F and H to the functions `time_update` and `measurement_update` respectively.
```julia
using KalmanFilter
Δt = 0.1
F = [1 Δt Δt^2/2; 0 1 Δt; 0 0 1]
H = [1, 0, 0]'
Q = [Δt^2/2; Δt; 1] * [Δt^2/2 Δt 1] * σ_acc_noise^2
R = σ_meas_noise^2
x_init = [0.0, 0.0, 0.0]
P_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
tu = time_update(x_init, P_init, F, Q)
for i = 1:100
    # Take a measurement
    mu = measurement_update(state(tu), covariance(tu), measurement, H, R)
    tu = time_update(state(mu), covariance(mu), F, Q)
end
```
### Non-linear case
If you pass a function instead of matrix for `F` or `H`, the Unscented Kalman Filter will be used.
```julia
F(x) = x .* [1., 2.]
tu = time_update(x, P, F, Q)
```
This can be inter changed meaning time update can use the usual linear Kalman Filter by passing a matrix and measurement update can use the Unscented Kalman Filter by passing a function or vise versa.

#### Augmentation
KalmanFilter also allows augmenting the noise covariances:
```julia
F(x, noise) = x .* [1., 2.] .+ noise
tu = time_update(x, P, F, Augment(Q))
H(x, noise) = x .* [1., 1.] .+ noise
mu = measurement_update(x, P, measurement, H, Augment(R))
```

### Statistical consistency testing
Goal: perform two of the most relevant consistency statistical tests of the Kalman filter
- the Normalized innovation squared (NIS) test
  - tests for unbiasedness
- the Innovation magnitude bound test
  - tests if approximately 68% (95%) of the innovation sequence values lie within the ⨦σ (⨦2σ) bound

## License

MIT License
