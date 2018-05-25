using Base.Test, KalmanFilter

srand(1234)

# Testparameters
const 𝐱 = [0, 1]
const 𝐲 = [0, 1]
const 𝐏 = diagm([2, 3])
const scales = ScalingParameters(1e-3, 2, 0)
const 𝐟(x) = x
const 𝐡(x) = x
const 𝐇 = eye(2)
const 𝐅 = eye(2)
const 𝐐 = eye(2)
const 𝐑 = eye(2)
const used_states = trues(2)

include("ukf.jl")
include("kf.jl")

@testset "Filter states" begin
    curr_used_states = [true, false]
    part_𝐱, part_𝐏 = KalmanFilter.filter_states(𝐱, 𝐏, curr_used_states)
    @test part_𝐱 == [0]
    @test part_𝐏 == 𝐏[curr_used_states, curr_used_states]
end

@testset "Expand states" begin
    𝐱_init = [0, 1]
    𝐏_init = diagm([2, 3])
    𝐱_prev = [1, 2]
    𝐏_prev = diagm([3, 4])
    part_𝐱 = [3]
    curr_used_states = [true, false]
    part_𝐏 = ones(2,2)[curr_used_states,curr_used_states] * 5
    reset_unused_states = false
    𝐱_expanded, 𝐏_expanded = KalmanFilter.expand_states(part_𝐱, part_𝐏, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, curr_used_states, reset_unused_states)
    @test 𝐱_expanded == [3,2]
    @test 𝐏_expanded == diagm([5,4])

    reset_unused_states = true
    𝐱_expanded, 𝐏_expanded = KalmanFilter.expand_states(part_𝐱, part_𝐏, 𝐱_init, 𝐏_init, 𝐱_prev, 𝐏_prev, curr_used_states, reset_unused_states)
    @test 𝐱_expanded == [3,1]
    @test 𝐏_expanded == diagm([5,3])
end