λ(scales::ScalingParameters, n) = scales.α^2 * (n + scales.κ) - n
mean_weights(scales::ScalingParameters, n) = [λ(scales, n) / (n + λ(scales, n)); fill(1 / (2 * (n + λ(scales, n))), 2 * n)]
cov_weights(scales::ScalingParameters, n) = [λ(scales, n) / (n + λ(scales, n)) + 1 - scales.α^2 + scales.β; fill(1 / (2 * (n + λ(scales, n))), 2 * n)]

"""
$(SIGNATURES)

Calculate Sigma Points.
`scales` is of type ScalingParameters.
"""
function calc_sigma_points(𝐱, 𝐏, scales)
    𝐏_chol = sqrt(length(𝐱) + λ(scales, length(𝐱))) * chol(Symmetric(𝐏))'
    [𝐱 𝐱 .+ 𝐏_chol 𝐱 .- 𝐏_chol]
end

"""
$(SIGNATURES)

UKF time update.
Returns the time updated Sigma Points, the time updated states and the time updated covariance.
"""
function _time_update(χ, scales, 𝐟::Function, 𝐐)
    num_states = size(χ, 1)
    χ_next = mapslices(𝐟, χ, 1)
    𝐱_next = χ_next * mean_weights(scales, num_states)
    𝐏_next = (χ_next .- 𝐱_next) .* cov_weights(scales, num_states)' * (χ_next .- 𝐱_next)' + 𝐐
    χ_next, 𝐱_next, 𝐏_next
end

"""
$(SIGNATURES)

UKF measurement update.
Returns the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function _measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, h::Function, 𝐑)
    𝓨 = mapslices(h, χ, 1)
    num_states = size(χ, 1)
    𝐲̂ = 𝓨 * mean_weights(scales, num_states)
    𝐲̃ = 𝐲 - 𝐲̂ # Innovation
    𝐏yy = (𝓨 .- 𝐲̂) .* cov_weights(scales, num_states)' * (𝓨 .- 𝐲̂)' + 𝐑 # Innovation covariance
    𝐏xy = (χ[1:length(𝐱),:] .- 𝐱) .* cov_weights(scales, num_states)' * (𝓨 .- 𝐲̂)' # Cross covariance
    𝐊 = 𝐏xy / 𝐏yy # Kalman gain
    𝐱_next = 𝐱 + 𝐊 * 𝐲̃
    𝐏_next = 𝐏 - 𝐊 * 𝐏yy * 𝐊'
    𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

"""
$(SIGNATURES)

UKF time update.
The transition noise covariance `𝐐` is NOT augmented.
Returns a measurement update function.
"""
function time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, scales::ScalingParameters, 𝐟::Function, 𝐐, used_states, reset_unused_states)
    part_𝐱, part_𝐏 = filter_states(𝐱, 𝐏, used_states)
    χ = calc_sigma_points(𝐱, 𝐏, scales)
    χ_next, 𝐱_next, 𝐏_next = _time_update(χ, scales, 𝐟, 𝐐)
    (𝐲, 𝐇, 𝐑) -> measurement_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, χ_next, 𝐱_next, 𝐏_next, scales, 𝐲, 𝐇, 𝐑, used_states, reset_unused_states)
end

"""
$(SIGNATURES)

UKF time update.
The transition noise covariance `𝐐` is augmented.
Returns a measurement update function.
"""
function time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, scales::ScalingParameters, 𝐟::Function, 𝐐::Augment, used_states, reset_unused_states)
    part_𝐱, part_𝐏 = filter_states(𝐱, 𝐏, used_states)
    part_𝐱ᵃ, part_𝐏ᵃ = augment(part_𝐱, part_𝐏, 𝐐)
    χᵃ = calc_sigma_points(part_𝐱ᵃ, part_𝐏ᵃ, scales)
    χ_next, 𝐱_next, 𝐏_next = _time_update(χᵃ, scales, 𝐟, 0)
    (𝐲, 𝐇, 𝐑) -> measurement_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, χ_next, 𝐱_next, 𝐏_next, scales, 𝐲, 𝐇, 𝐑, used_states, reset_unused_states)
end

"""
$(SIGNATURES)

UKF time update.
The transition noise covariance `𝐐` and the measurement noise covariance `𝐑` are augmented.
Returns a measurement update function.
"""
function time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, scales::ScalingParameters, 𝐟::Function, 𝐐::Augment, 𝐑::Augment, used_states, reset_unused_states)
    part_𝐱, part_𝐏 = filter_states(𝐱, 𝐏, used_states)
    part_𝐱ᵃ, part_𝐏ᵃ = augment(part_𝐱, part_𝐏, 𝐐, 𝐑)
    χᵃ = calc_sigma_points(part_𝐱ᵃ, part_𝐏ᵃ, scales)
    χ_next, 𝐱_next, 𝐏_next = _time_update(χᵃ, scales, 𝐟, 0)
    (𝐲, 𝐇) -> measurement_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, χ_next, 𝐱_next, 𝐏_next, scales, 𝐲, 𝐇, used_states, reset_unused_states)
end

"""
$(SIGNATURES)

UKF measurement update.
The time update was of type UKF.
The measurement noise covariance `𝐑` is already augmented in the time update augmented.
Returns a time update function, the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, χ, 𝐱, 𝐏, scales::ScalingParameters, 𝐲, h::Function, used_states, reset_unused_states)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, h, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, 𝐑, used_states::BitArray{1} = trues(length(𝐱))) ->
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, 𝐑, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

"""
$(SIGNATURES)

UKF measurement update.
The time update was of type UKF.
The measurement noise covariance `𝐑` is NOT augmented.
Returns a time update function, the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, χ, 𝐱, 𝐏, scales::ScalingParameters, 𝐲, h::Function, 𝐑, used_states, reset_unused_states)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, h, 𝐑)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, used_states::BitArray{1} = trues(length(𝐱))) ->
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

"""
$(SIGNATURES)

KF measurement update.
The time update was of type UKF.
The measurement noise covariance `𝐑` is already augmented in the time update.
Returns a time update function, the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, χ, 𝐱, 𝐏, scales::ScalingParameters, 𝐲, 𝐇, used_states, reset_unused_states)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(𝐱, 𝐏, 𝐲, 𝐇, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, 𝐑, used_states::BitArray{1} = trues(length(𝐱))) ->
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, 𝐑, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

"""
$(SIGNATURES)

KF measurement update.
The time update was of type UKF.
The measurement noise covariance `𝐑` is NOT augmented.
Returns a time update function, the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, χ, 𝐱, 𝐏, scales::ScalingParameters, 𝐲, 𝐇, 𝐑, used_states, reset_unused_states)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(𝐱, 𝐏, 𝐲, 𝐇, 𝐑)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, used_states::BitArray{1} = trues(length(𝐱))) ->
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

"""
$(SIGNATURES)

KF measurement update.
The time update was of type UKF.
The measurement noise covariance `𝐑` is augmented.
Returns a time update function, the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, χ, 𝐱, 𝐏, scales::ScalingParameters, 𝐲, 𝐇, 𝐑::Augment, used_states, reset_unused_states)
    𝐱ᵃ, 𝐏ᵃ = augment(𝐱, 𝐏, 𝐑)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(𝐱ᵃ, 𝐏ᵃ, 𝐲, 𝐇, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, used_states::BitArray{1} = trues(length(𝐱))) ->
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

"""
$(SIGNATURES)

UKF measurement update.
The time update was of type KF.
The measurement noise covariance `𝐑` is already augmented by the time update.
Returns a time update function, the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, 𝐱, 𝐏, scales::ScalingParameters, 𝐲, h::Function, used_states, reset_unused_states)
    χ = calc_sigma_points(𝐱, 𝐏, scales)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, h, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, 𝐑, used_states::BitArray{1} = trues(length(𝐱))) ->
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, 𝐑, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

"""
$(SIGNATURES)

UKF measurement update.
The time update was of type KF.
The measurement noise covariance `𝐑` is NOT augmented.
Returns a time update function, the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, 𝐱, 𝐏, scales::ScalingParameters, 𝐲, h::Function, 𝐑, used_states, reset_unused_states)
    χ = calc_sigma_points(𝐱, 𝐏, scales)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, h::Function, 𝐑)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, used_states::BitArray{1} = trues(length(𝐱))) ->
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

"""
$(SIGNATURES)

UKF measurement update.
The time update was of type KF.
The measurement noise covariance `𝐑` is augmented.
Returns a time update function, the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, 𝐱, 𝐏, scales::ScalingParameters, 𝐲, h::Function, 𝐑::Augment, used_states, reset_unused_states)
    𝐱ᵃ, 𝐏ᵃ = augment(𝐱, 𝐏, 𝐑)
    χ = calc_sigma_points(𝐱ᵃ, 𝐏ᵃ, scales)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, h, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, used_states::BitArray{1} = trues(length(𝐱))) ->
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end