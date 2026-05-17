using Test
using HyperbolicPDEs

println("Starting tests")
ti = time()

include("operators.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits = 2), " seconds")
