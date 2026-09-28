# Build a PR-comment markdown table from benchpkg result JSONs, reporting the
# *minimum* time per benchmark instead of AirspeedVelocity's median. The minimum
# is robust to noisy-neighbour contention on shared CI runners (it captures the
# least-disturbed sample), so it doesn't manufacture phantom regressions when the
# head build happens to land on a busier runner window.
#
# Usage:  julia bench_table.jl <input_dir> <pkg> <base_rev> <head_rev> [label]
# The optional <label> (e.g. the runner OS) is appended to the heading so a
# multi-platform matrix can post one distinct comment per platform.
using JSON3

input_dir, pkg, base_rev, head_rev = ARGS[1], ARGS[2], ARGS[3], ARGS[4]
label = length(ARGS) >= 5 ? ARGS[5] : ""

readrev(rev) = open(joinpath(input_dir, "results_$(pkg)@$(rev).json"), "r") do io
    JSON3.read(io, Dict{String,Any})
end

# Recursively collect leaf trials (nodes carrying a "times" array) keyed by "a/b/c".
function leaves!(acc, node, prefix = "")
    if haskey(node, "times")
        acc[prefix] = node
    elseif haskey(node, "data")
        for (k, v) in node["data"]
            name = isempty(prefix) ? String(k) : prefix * "/" * String(k)
            leaves!(acc, v, name)
        end
    end
    acc
end

mintime(node) = minimum(Float64.(node["times"]))   # ns

function fmt_time(ns)
    unit, div = ns < 1e3 ? ("ns", 1.0) :
                ns < 1e6 ? ("μs", 1e3) :
                ns < 1e9 ? ("ms", 1e6) : ("s", 1e9)
    string(round(ns / div; sigdigits = 3), " ", unit)
end

fmt_mem(node) = string(round(Int, node["allocs"]), " allocs: ", round(Int, node["memory"]), " B")

# Ratio cell base/head (>1 ⇒ PR faster / allocates less).
# ✅ ≥ 5 % better, ⚠️ ≥ 5 % worse.
function fmt_ratio(r)
    s = string(round(r; sigdigits = 3))
    r >= 1.05 ? "$s ✅" : r <= 0.95 ? "$s ⚠️" : s
end

# Memory ratio cell: base/head bytes allocated. The in-place variants usually
# allocate nothing, so guard the degenerate ratios: `∞` when the PR drops to zero
# allocations, `0` when it introduces them, `—` when both are zero.
function fmt_mem_ratio(b, h)
    bm = Float64(b["memory"]); hm = Float64(h["memory"])
    bm == 0 && hm == 0 && return "—"
    hm == 0 && return "∞ ✅"
    bm == 0 && return "0 ⚠️"
    fmt_ratio(bm / hm)
end

base = leaves!(Dict{String,Any}(), readrev(base_rev))
head = leaves!(Dict{String,Any}(), readrev(head_rev))

shortrev(rev) = length(rev) >= 16 && !occursin('/', rev) ? rev[1:8] * "…" : rev
headlbl = shortrev(head_rev)
baselbl = shortrev(base_rev)

# Stable ordering over the UNION of both revs' benchmarks (so rows new on the PR
# still appear). Sort alphabetically, then push time_to_load last.
names = sort(collect(union(keys(base), keys(head))))
filter!(n -> n != "time_to_load", names)
(haskey(base, "time_to_load") || haskey(head, "time_to_load")) && push!(names, "time_to_load")

io = IOBuffer()
suffix = isempty(label) ? "" : " — $label"
println(io, "## Benchmark Results (minimum time)$suffix")
println(io)
println(io, "Reporting the **minimum** over all samples (robust to shared-runner ",
            "contention), not the median.")
println(io)

println(io, "<details open><summary>Time benchmarks (base vs PR head)</summary>")
println(io)
println(io, "Ratio = $baselbl / $headlbl: **>1 means the PR is faster**. ✅ ≥ 5 % faster, ",
            "⚠️ ≥ 5 % slower. A blank cell means the benchmark exists on only one revision ",
            "(🆕 = new on the PR, 🗑 = removed).")
println(io)
println(io, "|  | $baselbl | $headlbl | $baselbl / $headlbl |")
println(io, "|:--|--:|--:|--:|")
for n in names
    b = get(base, n, nothing); h = get(head, n, nothing)
    bcell = b === nothing ? "" : fmt_time(mintime(b))
    hcell = h === nothing ? "" : fmt_time(mintime(h))
    if b !== nothing && h !== nothing
        rcell = fmt_ratio(mintime(b) / mintime(h))
    else
        rcell = h === nothing ? "🗑" : "🆕"
    end
    println(io, "| $n | $bcell | $hcell | $rcell |")
end
println(io)
println(io, "</details>")
println(io)

println(io, "<details><summary>Memory benchmarks (base vs PR head)</summary>")
println(io)
println(io, "Ratio = $baselbl / $headlbl (bytes allocated): **>1 means the PR allocates less**. ",
            "✅ ≥ 5 % less, ⚠️ ≥ 5 % more. `∞`/`0` mark a benchmark that drops to / picks up ",
            "allocations, `—` means both revisions allocate nothing. A blank cell means the ",
            "benchmark exists on only one revision (🆕 = new on the PR, 🗑 = removed).")
println(io)
println(io, "|  | $baselbl | $headlbl | $baselbl / $headlbl |")
println(io, "|:--|--:|--:|--:|")
for n in names
    b = get(base, n, nothing); h = get(head, n, nothing)
    bcell = b === nothing ? "" : fmt_mem(b)
    hcell = h === nothing ? "" : fmt_mem(h)
    if b !== nothing && h !== nothing
        rcell = fmt_mem_ratio(b, h)
    else
        rcell = h === nothing ? "🗑" : "🆕"
    end
    println(io, "| $n | $bcell | $hcell | $rcell |")
end
println(io)
println(io, "</details>")

print(String(take!(io)))
