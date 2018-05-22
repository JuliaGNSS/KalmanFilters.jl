module KalmanFilter

    using DocStringExtensions

    struct Augment{T}
        cov::T
    end

    struct ScalingParameters
        α::Float64
        β::Float64
        κ::Float64
    end

    export ScalingParameters, Augment, init_kalman

    include("kf.jl")
    include("ukf.jl")

    """
    $(SIGNATURES)

    Augment the state and covariance
    """
    function augment(𝐱, 𝐏, 𝐑::Augment)
        𝐱ᵃ = [𝐱; zeros(size(𝐑.cov, 1))]
        𝐏ᵃ = [𝐏                               zeros(size(𝐏,1),size(𝐑.cov,2));
              zeros(size(𝐑.cov,1),size(𝐏,2))  𝐑.cov                          ]
        𝐱ᵃ, 𝐏ᵃ
    end

    """
    $(SIGNATURES)

    Augment the state and covariance twice
    """
    function augment(𝐱, 𝐏, 𝐐, 𝐑)
        augment(augment(𝐱, 𝐏, 𝐐)..., 𝐑)
    end

    """
    $(SIGNATURES)

    Initialize Kalman Filter.
    `𝐱` is the initial state, `𝐏` is the initial covariance, `scales` is optional and holds the scaling
    parameters for the UKF and `reset_unused_states` is optional and declares if unused states should be 
    resetted to the initals.
    Returns the time update function. The time update function depends on the transition noise covariance
    matrix `𝐐`, which can be augmented by `Augment(𝐐)`, optionally on the measurement noise covariance
    matrix `Augment(𝐑)`, only if augmented, optionally on the used states `used_states` and on the
    transition, which can be of type scalar, Matrix or Function. In the latter case the transition is
    assumed to be non-linear and the Unscented Kalman Filter (UKF) is used instead of the Kalman Filter (KF).
    """
    function init_kalman(𝐱, 𝐏, scales = ScalingParameters(1, 2, 0), reset_unused_states = true)
        num_states = length(𝐱)
        𝐱_init = copy(𝐱)
        𝐏_init = copy(𝐏)
        rtn_time_update(𝐟_or_𝐓, 𝐐, used_states::BitArray{1} = trues(num_states)) = 
            time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, scales, 𝐟_or_𝐓, 𝐐, used_states, reset_unused_states)
        rtn_time_update(𝐟_or_𝐓, 𝐐, 𝐑, used_states::BitArray{1} = trues(num_states)) = 
            time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, scales, 𝐟_or_𝐓, 𝐐, 𝐑, used_states, reset_unused_states)
        rtn_time_update
    end

    """
    $(SIGNATURES)

    Filter the state and the covariance based on the current used states.
    """
    function filter_states(𝐱, 𝐏, used_states)
        part_𝐱 = 𝐱[used_states]
        part_𝐏 = 𝐏[used_states, used_states]
        part_𝐱, part_𝐏
    end

    """
    $(SIGNATURES)

    Updates the previous states with the filtered updated states.
    """
    function expand_states(part_𝐱, part_𝐏, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
        num_states = length(used_states)
        num_used_states = sum(used_states)
        𝐱 = zeros(num_states)
        𝐱[used_states] = part_𝐱[1:num_used_states]
        𝐱 = reset_unused_states ? 𝐱_init .* .!used_states .+ 𝐱 : 𝐱_prev .* .!used_states .+ 𝐱
        𝐏 = zeros(num_states, num_states)
        𝐏[used_states, used_states] = part_𝐏[1:num_used_states, 1:num_used_states]
        𝐏 = reset_unused_states ? 𝐏_init .* .!(used_states * used_states') .+ 𝐏 : 𝐏_prev .* .!(used_states * used_states') .+ 𝐏
        𝐱, 𝐏
    end

end
