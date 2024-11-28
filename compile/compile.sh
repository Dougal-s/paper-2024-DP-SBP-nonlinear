#!/bin/bash

cd "$(dirname "$0")"

mkdir -p tmp

julia -e '
using PackageCompiler
using DrWatson
@quickactivate

PackageCompiler.create_sysimage( [ "Makie"
                                 , "OrdinaryDiffEqSSPRK"
                                 , "SummationByPartsOperators"
                                 , "Symbolics"
								 ]
                               ; sysimage_path="tmp/sysimage.so"
                               , precompile_execution_file=["symbolics_example.jl", "makie_example.jl", "example.jl"]
                               , sysimage_build_args=`--heap-size-hint 75%`
                               )
'

cp tmp/sysimage.so sysimage.so
