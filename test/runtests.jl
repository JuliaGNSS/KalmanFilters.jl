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
include("allocations.jl")
include("inplace.jl")
include("static.jl")

# That the static filters don't allocate is only checked from Julia 1.12 on. It needs the
# compiler to drop the bounds checks of their mutable copies (see `static_allocations.jl`),
# so with forced bounds checks, as on CI, the check runs in a process of its own.
if VERSION >= v"1.12"
    if Base.JLOptions().check_bounds == 1
        file = joinpath(@__DIR__, "static_allocations.jl")
        julia = Base.julia_cmd()
        project = Base.active_project()
        cmd = `$julia --check-bounds=auto --project=$project $file`
        @test success(pipeline(cmd; stdout, stderr))
    else
        include("static_allocations.jl")
    end
end
