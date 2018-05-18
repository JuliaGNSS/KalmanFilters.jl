struct ScalingParameters
    α::Float64
    β::Float64
    κ::Float64
end

λ(scales::ScalingParameters, n) = scales.α^2 * (n + scales.κ) - n
mean_weights(scales::ScalingParameters, n) = [λ(scales, n) / (n + λ(scales, n)); fill(1 / (2 * (n + λ(scales, n))), 2 * n)]
cov_weights(scales::ScalingParameters, n) = [λ(scales, n) / (n + λ(scales, n)) + 1 - scales.α^2 + scales.β; fill(1 / (2 * (n + λ(scales, n))), 2 * n)]

function calc_sigma_points(𝐱, 𝐏, scales)
    𝐏_chol = sqrt(length(𝐱) + λ(scales, length(𝐱))) * chol(Symmetric(𝐏))'
    [𝐱 𝐱 .+ 𝐏_chol 𝐱 .- 𝐏_chol]
end

function _time_update(𝐱, 𝐏, scales, 𝐟::Function, 𝐐)
    χ = calc_sigma_points(𝐱, 𝐏, scales)
    num_states = size(χ, 1)
    χ_next = mapslices(𝐟, χ, 1)
    𝐱_next = χ_next * mean_weights(scales, num_states)
    𝐏_next = (χ_next - 𝐱_next) .* cov_weights(scales, num_states)' * (χ_next - 𝐱_next)' + 𝐐
    χ_next, 𝐱_next, 𝐏_next
end

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

function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, χ, 𝐱, 𝐏, scales, 𝐲, h::Function, used_states, reset_unused_states)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, h, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, 𝐑, used_states::BitArray{1} = trues(length(𝐱))) -> 
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, 𝐑, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, χ, 𝐱, 𝐏, scales, 𝐲, h::Function, 𝐑, used_states, reset_unused_states)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, h, 𝐑)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, used_states::BitArray{1} = trues(length(𝐱))) -> 
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, 𝐱, 𝐏, scales, 𝐲, h::Function, used_states, reset_unused_states)
    χ = calc_sigma_points(𝐱, 𝐏, scales)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, h, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, 𝐑, used_states::BitArray{1} = trues(length(𝐱))) -> 
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, 𝐑, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, 𝐱, 𝐏, scales, 𝐲, h::Function, 𝐑, used_states, reset_unused_states)
    χ = calc_sigma_points(𝐱, 𝐏, scales)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, h::Function, 𝐑)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, used_states::BitArray{1} = trues(length(𝐱))) -> 
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, 𝐱, 𝐏, scales, 𝐲, h::Function, 𝐑::Augment, used_states, reset_unused_states)
    𝐱ᵃ, 𝐏ᵃ = augment(𝐱, 𝐏, 𝐑)
    χ = calc_sigma_points(𝐱ᵃ, 𝐏ᵃ, scales)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, h, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, used_states::BitArray{1} = trues(length(𝐱))) -> 
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, scales, 𝐓, 𝐐, used_states, reset_unused_states), 𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end