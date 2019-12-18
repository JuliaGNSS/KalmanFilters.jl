using Test, KalmanFilter, Random, LinearAlgebra

Random.seed!(1234)

include("ukf.jl")
include("srukf.jl")
include("sraukf.jl")
include("kf.jl")
include("srkf.jl")
include("aukf.jl")
include("tests.jl")
include("system.jl")
