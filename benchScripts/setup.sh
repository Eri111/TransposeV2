#!/usr/bin/env bash
name=hello

mkdir ../bins/$name

cp ../build/memory-benchmark-media ../bins/$name

cp ../benchmarks/memory-benchmark.cpp ../bins/$name
