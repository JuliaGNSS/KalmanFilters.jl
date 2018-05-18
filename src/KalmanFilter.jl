module KalmanFilter

    struct Augment{T}
        cov::T
    end

    export Weights, Augment, init_kalman

    include("kf.jl")
    include("ukf.jl")

    function augment(𝐱, 𝐏, 𝐑::Augment)
        𝐱ᵃ = [𝐱; zeros(size(𝐑.cov, 1))]
        𝐏ᵃ = [𝐏                               zeros(size(𝐏,1),size(𝐑.cov,2));
              zeros(size(𝐑.cov,1),size(𝐏,2))  𝐑.cov                          ]
        𝐱ᵃ, 𝐏ᵃ
    end

    function augment(𝐱, 𝐏, 𝐐, 𝐑)
        augment(augment(𝐱, 𝐏, 𝐐)..., 𝐑)
    end

    function init_kalman(𝐱, 𝐏, weights = Weights(1, 2, 0), reset_unused_states = true)
        num_states = length(𝐱)
        𝐱_init = copy(𝐱)
        𝐏_init = copy(𝐏)
        rtn_time_update(𝐟_or_𝐓, 𝐐, used_states::BitArray{1} = trues(num_states)) = 
            time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, weights, 𝐟_or_𝐓, 𝐐, used_states, reset_unused_states)
        rtn_time_update(𝐟_or_𝐓, 𝐐, 𝐑, used_states::BitArray{1} = trues(num_states)) = 
            time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, weights, 𝐟_or_𝐓, 𝐐, 𝐑, used_states, reset_unused_states)
        rtn_time_update
    end

    function time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, weights, 𝐟_or_𝐓, 𝐐, used_states, reset_unused_states)
        part_𝐱, part_𝐏 = filter_states(𝐱, 𝐏, used_states)
        iweights = InternalWeights(weights, sum(used_states))
        time_update_output = _time_update(part_𝐱, part_𝐏, iweights, 𝐟_or_𝐓, 𝐐)
        (𝐲, 𝐇, 𝐑) -> measurement_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, time_update_output..., iweights, weights, 𝐲, 𝐇, 𝐑, used_states, reset_unused_states)
    end
    
    function time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, weights, 𝐟_or_𝐓, 𝐐::Augment, used_states, reset_unused_states)
        part_𝐱, part_𝐏 = filter_states(𝐱, 𝐏, used_states)
        part_𝐱ᵃ, part_𝐏ᵃ = augment(part_𝐱, part_𝐏, 𝐐)
        iweights = InternalWeights(weights, sum(used_states))
        time_update_output = _time_update(part_𝐱ᵃ, part_𝐏ᵃ, iweights, 𝐟_or_𝐓, 0)
        (𝐲, 𝐇, 𝐑) -> measurement_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, time_update_output..., iweights, weights, 𝐲, 𝐇, 𝐑, used_states, reset_unused_states)
    end
    
    function time_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, weights, 𝐟_or_𝐓, 𝐐::Augment, 𝐑::Augment, used_states, reset_unused_states)
        part_𝐱, part_𝐏 = filter_states(𝐱, 𝐏, used_states)
        part_𝐱ᵃ, part_𝐏ᵃ = augment(part_𝐱, part_𝐏, 𝐐, 𝐑)
        iweights = InternalWeights(weights, sum(used_states))
        time_update_output = _time_update(part_𝐱ᵃ, part_𝐏ᵃ, iweights, 𝐟_or_𝐓, 0)
        (𝐲, 𝐇) -> measurement_update(𝐱_init, 𝐏_init, 𝐱, 𝐏, time_update_output..., iweights, weights, 𝐲, 𝐇, used_states, reset_unused_states)
    end

    function filter_states(𝐱, 𝐏, used_states)
        part_𝐱 = 𝐱[used_states]
        part_𝐏 = 𝐏[used_states, used_states]
        part_𝐱, part_𝐏
    end

    function expand_states(part_𝐱, part_𝐏, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, used_states, reset_unused_states)
        num_states = length(used_states)
        𝐱 = zeros(num_states)
        𝐱[used_states] = part_𝐱
        𝐱 = reset_unused_states ? 𝐱_init .* .!used_states .+ 𝐱 : 𝐱_prev .* .!used_states .+ 𝐱
        𝐏 = zeros(num_states, num_states)
        𝐏[used_states, used_states] = part_𝐏
        𝐏 = reset_unused_states ? 𝐏_init .* .!(used_states * used_states') .+ 𝐏 : 𝐏_prev .* .!(used_states * used_states') .+ 𝐏
        𝐱, 𝐏
    end

end
