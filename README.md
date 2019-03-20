
[![pipeline status](https://git.rwth-aachen.de/nav/KalmanFilter.jl/badges/master/pipeline.svg)](https://git.rwth-aachen.de/nav/KalmanFilter.jl/commits/master)
[![coverage report](https://git.rwth-aachen.de/nav/KalmanFilter.jl/badges/master/coverage.svg)](https://git.rwth-aachen.de/nav/KalmanFilter.jl/commits/master)
# KalmanFilter
Provides multiple Kalman Filters like KF, AKF, UKF, AUKF

## Getting started

Install:
```julia
julia> ]
pkg> add git@git.rwth-aachen.de:nav/KalmanFilter.jl.git
```

## Usage

### Linear case
If matrices 𝐅 and 𝐇 are passed to time update and measurement update respectively, the usual Kalman Filter will be used:
```julia
using KalmanFilter
Δt = 0.1
F = [1 Δt Δt^2/2; 0 1 Δt; 0 0 1]
H = [1, 0, 0]'
Q = [Δt^2/2; Δt; 1] * [Δt^2/2 Δt 1] * σ_acc_noise^2
R = σ_meas_noise^2
x_init = [0.0, 0.0, 0.0]
P_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
kalman_inits = KalmanInits(x_init, P_init)
time_update_results = time_update(kalman_inits, F, Q)
for i = 1:maxiter
    # Take a measurement
    measurement_update_results = measurement_update(time_update_results, measurement, H, R)
    time_update_results = time_update(measurement_update_results, F, Q)
end
```
### Non-linear case
If you pass a function instead of matrix for `F` or `H`, the Unscented Kalman Filter will be used.
This can be inter changed. This means time update can use the usual linear Kalman Filter by passing a matrix and measurement update can use the Unscented Kalman Filter by passing a function or vise versa.

### Augmentation
KalmanFilter.jl also allows augmenting the noise covariances:
```julia
time_update_results = time_update(kalman_inits, F, Augment(Q))
```
or
```julia
measurement_update_results = measurement_update(time_update_results, measurement, H, Augment(R))
```

### Statistical consistency testing
Goal: perform two of the most relevant consistency statistical tests of the Kalman filter
- the Normalized innovation squared (NIS) test
  - tests for unbiasedness
- the Innovation magnitude bound test
  - tests if approximately 68% (95%) of the innovation sequence values lie within the ⨦σ (⨦2σ) bound

## License

MIT License
