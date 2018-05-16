struct InternalWeights
    λ::Float64
    m::Vector{Float64}  # Mean weight
    c::Vector{Float64}  # Covariance weight
end

struct Weights
    α::Float64
    β::Float64
    κ::Float64
end

function InternalWeights(weights, n)
    λ = weights.α^2 * (n + weights.κ) - n
    weight_mean = [λ / (n + λ); fill(1 / (2 * (n + λ)), 2 * n)]
    weight_cov = [λ / (n + λ) + 1 - weights.α^2 + weights.β; fill(1 / (2 * (n + λ)), 2 * n)]
    InternalWeights(λ, weight_mean, weight_cov)
end

function calc_sigma_points(𝐱, 𝐏, iweights)
    𝐏_chol = sqrt(length(𝐱) + iweights.λ) * chol(𝐏)'
    [𝐱 𝐱 .+ 𝐏_chol 𝐱 .- 𝐏_chol]
end

function _time_update(𝐱, 𝐏, iweights, 𝐟::Function, 𝐐)
    χ = calc_sigma_points(𝐱, 𝐏, iweights)
    χ_next = mapslices(𝐟, χ, 1)
    𝐱_next = χ_next * iweights.m
    𝐏_next = (χ_next - 𝐱_next) .* iweights.c' * (χ_next - 𝐱_next)' + 𝐐
    χ_next, 𝐱_next, 𝐏_next
end

function _measurement_update(χ, 𝐱, 𝐏, iweights, 𝐲, h::Function, 𝐑)
    𝓨 = mapslices(h, χ, 1)
    𝐲̂ = 𝓨 * iweights.m
    𝐲̃ = 𝐲 - 𝐲̂ # Innovation
    𝐏yy = (𝓨 - 𝐲̂) .* iweights.c' * (𝓨 - 𝐲̂)' + 𝐑 # Innovation covariance
    𝐏xy = (χ - 𝐱) .* iweights.c' * (𝓨 - 𝐲̂)' # Cross covariance
    𝐊 = 𝐏xy / 𝐏yy # Kalman gain
    𝐱_next = 𝐱 + 𝐊 * 𝐲̃
    𝐏_next = 𝐏 - 𝐊 * 𝐏yy * 𝐊'
    𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy
end

function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, χ, 𝐱, 𝐏, iweights, 𝐲, h, used_states, reset_unused_states)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, iweights, 𝐲, h, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, 𝐑, used_states = trues(length(𝐱))) -> 
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, iweights, 𝐓, 𝐐, 𝐑, used_states, reset_unused_states), part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy
end

function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, χ, 𝐱, 𝐏, iweights, 𝐲, h, 𝐑::Matrix, used_states, reset_unused_states)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, iweights, 𝐲, h, 𝐑)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, used_states = trues(length(𝐱))) -> 
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, iweights, 𝐓, 𝐐, used_states, reset_unused_states), part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy
end

function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, 𝐱, 𝐏, iweights, 𝐲, h, used_states, reset_unused_states)
    χ = calc_sigma_points(𝐱, 𝐏, iweights)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, iweights, 𝐲, h, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, 𝐑, used_states = trues(length(𝐱))) -> 
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, iweights, 𝐓, 𝐐, 𝐑, used_states, reset_unused_states), part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy
end

function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, 𝐱, 𝐏, iweights, 𝐲, h, 𝐑::Matrix, used_states, reset_unused_states)
    χ = calc_sigma_points(𝐱, 𝐏, iweights)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, iweights, 𝐲, h, 𝐑)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, used_states = trues(length(𝐱))) -> 
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, iweights, 𝐓, 𝐐, used_states, reset_unused_states), part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy
end

function measurement_update(𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, 𝐱, 𝐏, iweights, 𝐲, h, 𝐑::Augment, used_states, reset_unused_states)
    𝐱ᵃ, 𝐏ᵃ = augment(𝐱, 𝐏, 𝐑)
    χ = calc_sigma_points(𝐱ᵃ, 𝐏ᵃ, iweights)
    part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy = _measurement_update(χ, 𝐱, 𝐏, iweights, 𝐲, h, 0)
    𝐱_next, 𝐏_next = expand_states(part_𝐱_next, part_𝐏_next, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
    (𝐓, 𝐐, used_states = trues(length(𝐱))) -> 
        time_update(𝐱_init, 𝐏_init, 𝐱_next, 𝐏_next, iweights, 𝐓, 𝐐, used_states, reset_unused_states), part_𝐱_next, part_𝐏_next, 𝐲̂, 𝐏yy
end