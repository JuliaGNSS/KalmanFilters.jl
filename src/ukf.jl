struct Weights
    λ::Float64
    m::Vector{Float64}  # Mean weight
    c::Vector{Float64}  # Covariance weight
end

function Weights(α, β, κ, n)
    λ = α^2 * (n + κ) - n
    weight_mean = [λ / (n + λ); fill(1 / (2 * (n + λ)), 2 * n)]
    weight_cov = [λ / (n + λ) + 1 - α^2 + β; fill(1 / (2 * (n + λ)), 2 * n)]
    Weights(λ, weight_mean, weight_cov)
end

function calc_sigma_points(x, P, weights)
    P_chol = sqrt(length(x) + weights.λ) * chol(P)'
    [x x .+ P_chol x .- P_chol]
end

function ukf_time_update(χ, f, weights)
    χ_next = mapslices(f, χ, 1)
    x_next = χ_next * weights.m
    P_next = (χ_next - x_next) .* weights.c' * (χ_next - x_next)'
    χ_next, x_next, P_next
end

function ukf_time_update(χ, f, Q, weights)
    χ_next, x_next, P_next = time_update(χ, f, weights)
    χ_next, x_next, P_next + Q
end

function ukf_measurement_update(χ, x, P, h, y, weights)
    𝓨 = mapslices(h, χ, 1)
    ŷ = 𝓨 * weights.m
    ỹ = y - ŷ # Innovation
    Pyy = (𝓨 - ŷ) .* weights.c' * (𝓨 - ŷ)' # Innovation covariance
    Pxy = (χ - x) .* weights.c' * (𝓨 - ŷ)' # Cross covariance
    K = Pxy / Pyy # Kalman gain
    x_next = x + K * ỹ
    P_next = P - K * Pyy * K'
    x_next, P_next, ỹ, Pyy
end