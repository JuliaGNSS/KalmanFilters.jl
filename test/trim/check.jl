# Builds `app.jl` into a `juliac --trim=safe` executable and checks that it prints
# exactly what the same app prints in a regular Julia session. Needs Julia 1.12+ and
# the JuliaC app (`pkg> app add JuliaC`):
#
#     julia --project=test/trim -e 'using Pkg; Pkg.instantiate()'
#     julia --project=test/trim test/trim/check.jl
#
# The trim verifier rejects any call it cannot resolve statically, so a build failure
# here names the dynamic dispatch that crept into a filter.
juliac = something(
    Sys.which("juliac"),
    joinpath(first(DEPOT_PATH), "bin", Sys.iswindows() ? "juliac.bat" : "juliac"),
)
isfile(juliac) || error("juliac not found; install it with `pkg> app add JuliaC`")

app = joinpath(@__DIR__, "app.jl")
build = mktempdir()
run(`$juliac --output-exe kalman_filters_app --project $(@__DIR__) --bundle $build
    --trim=safe --experimental $app`)

executable = joinpath(build, "bin", Sys.iswindows() ? "kalman_filters_app.exe" : "kalman_filters_app")
trimmed = read(`$executable`, String)
regular = read(`$(Base.julia_cmd()) --project=$(@__DIR__) $app`, String)

print(trimmed)
if trimmed != regular
    println("\nThe regular session printed instead:\n", regular)
    error("the trimmed executable's output differs from the regular session's")
end
println("Trimmed executable ($(filesize(executable) ÷ 1024) KiB) matches the regular ",
    "session on all $(count(==('\n'), trimmed)) updates.")
