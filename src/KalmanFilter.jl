module KalmanFilter

    export calc_weights, calc_sigma_points, time_update, augment, measurement_update, time_update_linear

    struct Weights
        λ::Float64
        m::Vector{Float64}  # Mean weight
        c::Vector{Float64}  # Covariance weight
    end

    function calc_weights(α, β, κ, n)
        λ = α^2 * (n + κ) - n
        weight_mean = [λ / (n + λ); fill(1 / (2 * (n + λ)), 2 * n)]
        weight_cov = [λ / (n + λ) + 1 - α^2 + β; fill(1 / (2 * (n + λ)), 2 * n)]
        Weights(λ, weight_mean, weight_cov)
    end

    function calc_sigma_points(x, P, n, weights)
        √ = sqrt(n + weights.λ) * chol(P)'
        [x x + √ x - √]
    end

    function time_update(χ, f, weights)
        χ_next = mapslices(f, χ, 1)
        x_next = χ_next * weights.m
        P_next = (χ_next - x_next) .* weights.c' * (χ_next - x_next)'
        χ_next, x_next, P_next
    end

    function time_update(χ, f, Q, weights)
        χ_next, x_next, P_next = time_update(χ, f, weights)
        χ_next, x_next, P_next + Q
    end

    function time_update_linear(x, P, T, Q)
        T * x, T * P * T' + Q
    end

    function augment(x, P, R)
        x_a = [x; zeros(size(R, 1))]
        P_a = [P                            zeros(size(P,1),size(R,2));
               zeros(size(R,1),size(P,2))   R ]
        x_a, P_a
    end

    function augment(x, P, Q, R)
        augment(augment(x, P, Q)..., R)
    end

    function measurement_update(χ, x, P, h, y, weights)
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

end
