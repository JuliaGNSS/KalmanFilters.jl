"""
$(SIGNATURES)

KF time update.
Returns the time updated states and the time updated covariance.
"""
function _time_update(𝐱, 𝐏, 𝐅, 𝐐)
    𝐱_next = 𝐅 * 𝐱
    𝐏_next = 𝐅 * 𝐏 * 𝐅' .+ 𝐐
    𝐱_next, 𝐏_next
end

"""
$(SIGNATURES)

UKF measurement update.
Returns the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function _measurement_update(𝐱, 𝐏, 𝐲, 𝐇, 𝐑)
    𝐲̃ = 𝐲 .- 𝐇 * 𝐱
    𝐒 = 𝐇 * 𝐏 * 𝐇' .+ 𝐑
    𝐊 = 𝐏 * 𝐇' / 𝐒
    𝐱_next = 𝐱 .+ 𝐊 * 𝐲̃
    𝐏_next = 𝐏 .- 𝐊 * 𝐒 * 𝐊'
    𝐱_next, 𝐏_next, 𝐲̃, 𝐒
end

"""
$(SIGNATURES)

KF time update.
The transition noise covariance `𝐐` is NOT augmented.
Returns a measurement update function.
"""
function time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, scales::ScalingParameters, 𝐅, 𝐐, used_states, reset_unused_states)
    part_𝐱, part_𝐏 = filter_states(𝐱, 𝐏, used_states)
    𝐱_next, 𝐏_next = _time_update(part_𝐱, part_𝐏, 𝐅, 𝐐)
    (𝐲, 𝐇, 𝐑) -> measurement_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, 𝐱_next, 𝐏_next, scales, 𝐲, 𝐇, 𝐑, used_states, reset_unused_states)
end

"""
$(SIGNATURES)

KF time update.
The transition noise covariance `𝐐` is augmented.
Returns a measurement update function.
"""
function time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, scales::ScalingParameters, 𝐅, 𝐐::Augment, used_states, reset_unused_states)
    part_𝐱, part_𝐏 = filter_states(𝐱, 𝐏, used_states)
    part_𝐱ᵃ, part_𝐏ᵃ = augment(part_𝐱, part_𝐏, 𝐐)
    𝐱_next, 𝐏_next = _time_update(part_𝐱ᵃ, part_𝐏ᵃ, 𝐅, 0)
    (𝐲, 𝐇, 𝐑) -> measurement_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, 𝐱_next, 𝐏_next, scales, 𝐲, 𝐇, 𝐑, used_states, reset_unused_states)
end

"""
$(SIGNATURES)

KF time update.
The transition noise covariance `𝐐` and the measurement noise covariance `𝐑` are augmented.
Returns a measurement update function.
"""
function time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, scales::ScalingParameters, 𝐅, 𝐐::Augment, 𝐑::Augment, used_states, reset_unused_states)
    part_𝐱, part_𝐏 = filter_states(𝐱, 𝐏, used_states)
    part_𝐱ᵃ, part_𝐏ᵃ = augment(part_𝐱, part_𝐏, 𝐐, 𝐑)
    𝐱_next, 𝐏_next = _time_update(part_𝐱ᵃ, part_𝐏ᵃ, 𝐅, 0)
    (𝐲, 𝐇) -> measurement_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, 𝐱_next, 𝐏_next, scales, 𝐲, 𝐇, used_states, reset_unused_states)
end

"""
$(SIGNATURES)

KF measurement update.
The time update was of type KF.
The measurement noise covariance `𝐑` is already augmented in the time update.
Returns a time update function, the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, 𝐱, 𝐏, scales::ScalingParameters, 𝐲, 𝐇, used_states, reset_unused_states)
    part_𝐱_next, part_𝐏_next, 𝐲̃, 𝐏yy = _measurement_update(𝐱, 𝐏, 𝐲, 𝐇, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐅, 𝐐, 𝐑, used_states = 1:length(𝐱)) ->
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐅, 𝐐, 𝐑, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy
end

"""
$(SIGNATURES)

KF measurement update.
The time update was of type KF.
The measurement noise covariance `𝐑` is NOT augmented.
Returns a time update function, the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, 𝐱, 𝐏, scales::ScalingParameters, 𝐲, 𝐇, 𝐑, used_states, reset_unused_states)
    part_𝐱_next, part_𝐏_next, 𝐲̃, 𝐏yy = _measurement_update(𝐱, 𝐏, 𝐲, 𝐇, 𝐑)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐅, 𝐐, used_states = 1:length(𝐱)) ->
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐅, 𝐐, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy
end

"""
$(SIGNATURES)

KF measurement update.
The time update was of type KF.
The measurement noise covariance `𝐑` is augmented.
Returns a time update function, the measurement updated states, the measurement updated covariance,
the innovation and the innovation covariance.
"""
function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, 𝐱, 𝐏, scales::ScalingParameters, 𝐲, 𝐇, 𝐑::Augment, used_states, reset_unused_states)
    𝐱ᵃ, 𝐏ᵃ = augment(𝐱, 𝐏, 𝐑)
    part_𝐱_next, part_𝐏_next, 𝐲̃, 𝐏yy = _measurement_update(𝐱ᵃ, 𝐏ᵃ, 𝐲, 𝐇, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐅, 𝐐, used_states = 1:length(𝐱)) ->
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐅, 𝐐, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy
end
