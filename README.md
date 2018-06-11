
[![pipeline status](https://git.rwth-aachen.de/nav/KalmanFilter.jl/badges/master/pipeline.svg)](https://git.rwth-aachen.de/nav/KalmanFilter.jl/commits/master)
[![coverage report](https://git.rwth-aachen.de/nav/KalmanFilter.jl/badges/master/coverage.svg)](https://git.rwth-aachen.de/nav/KalmanFilter.jl/commits/master)
# KalmanFilter
Provides multiple Kalman Filters like KF, AKF, UKF, AUKF

## Getting started

Install:
```julia
Pkg.clone("git@git.rwth-aachen.de:nav/KalmanFilter.jl.git")
```

## Usage

### Linear case
If matrices 𝐅 and 𝐇 are passed to time update and measurement update respectively, the usual Kalman Filter will be used:
```julia
using KalmanFilter
𝐱_init = [0, 1]
𝐏_init = diagm([2, 3])
𝐅 = [1 0.1; 0 1]
𝐐 = diagm([0.25, 0.25])
𝐇 = [1 0]
𝐑 = 0.1
𝐲 = 5
time_update = KalmanFilter.init_kalman(𝐱_init, 𝐏_init)
measurement_update = time_update(𝐅, 𝐐)
time_update, 𝐱, 𝐏 = measurement_update(𝐲, 𝐇, 𝐑)
```
### Non-linear case
If you pass functions instead, the Unscented Kalman Filter will be used:
```julia
using KalmanFilter
𝐱_init = [0, 1]
𝐏_init = diagm([2, 3])
𝐟(𝐱) = [𝐱[1] + 0.1 * 𝐱[2]; 𝐱[2]]
𝐐 = diagm([0.25, 0.25])
𝐡(𝐱) = 𝐱[1]
𝐑 = 0.1
𝐲 = 5
time_update = KalmanFilter.init_kalman(𝐱_init, 𝐏_init)
measurement_update = time_update(𝐟, 𝐐)
time_update, 𝐱, 𝐏 = measurement_update(𝐲, 𝐡, 𝐑)
```
This can be inter changed, meaning time update can use the usual Kalman Filter by passing a matrix and measurement update can use the Unscented Kalman Filter by passing a function or vise versa.

### Augmentation
KalmanFilter.jl also allows augmenting the noise covariances:
```julia
measurement_update = time_update(𝐅, 𝐐)
time_update, 𝐱, 𝐏 = measurement_update(5, 𝐇, Augment(𝐑))
```
or
```julia
measurement_update = time_update(𝐟, Augment(𝐐), Augment(𝐑))
time_update, 𝐱, 𝐏 = measurement_update(5, 𝐡)
```

### Statistical consistency testing
Goal: perform two of the most relevant consistency statistical tests of the Kalman filter
- the Normalized innovation squared (NIS) test
  - tests for unbiasedness
- the Innovation magnitude bound test
  - tests if approximately 68% (95%) of the innovation sequence values lie within the ⨦σ (⨦2σ) bound

```julia
using KalmanFilter
𝐱_init = [0, 1]
𝐏_init = diagm([2, 3])
𝐅 = [1 0.1; 0 1]
𝐐 = diagm([0.25, 0.25])
𝐇 = [1 0]
𝐑 = 0.1
time_update = KalmanFilter.init_kalman(𝐱_init, 𝐏_init)
𝐲̃_over_time = Vector(length(range))
𝐒_over_time = Vector{Matrix{Float64}}(length(range))
nis_over_time = Vector(length(range))
for i = 1:2000
    measurement_update = time_update(𝐅, 𝐐)
    𝐳 = measurement()
    time_update, 𝐱, 𝐏, 𝐲̃, 𝐒 = measurement_update(𝐳, 𝐇, 𝐑)
    𝐲̃_over_time[i] = 𝐲̃
    𝐒_over_time[i] = 𝐒
    nis_over_time[i] = nis(𝐲̃, 𝐒)
end

dof = length(nis_over_time) * size(𝐲̃_over_time[1], 1)
if nis_test(nis_over_time, dof)
  println("NIS-test passed. Yay!")
else
  println("NIS-test failed. Oh no!")
end
if sigma_bound_test(𝐲̃_over_time, 𝐒_over_time)
  println("Sigma bound test for ⨦σ bound passed. Yay!")
else
  println("Sigma bound test for ⨦σ bound failed. Oh no!")
end
if two_sigma_bound_test(𝐲̃_over_time, 𝐒_over_time)
  println("Sigma bound test for ⨦2σ bound passed. Yay!")
else
  println("Sigma bound test for ⨦2σ bound failed. Oh no!")
end
```

## License

MIT License
