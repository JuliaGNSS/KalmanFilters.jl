using Test, KalmanFilters, Random, LinearAlgebra, LazyArrays, Statistics, FFTW, StaticArrays

Random.seed!(1234)

include("kf.jl")
include("ekf.jl")
include("srkf.jl")
include("sigmapoints.jl")
include("augmented_sigmapoints.jl")
include("ukf.jl")
include("srukf.jl")
include("aukf.jl")
include("sraukf.jl")
include("consider.jl")
include("tests.jl")
include("system.jl")
