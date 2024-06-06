# BatchedTransformations

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://murrellgroup.github.io/BatchedTransformations.jl/dev/)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![Build Status](https://github.com/MurrellGroup/BatchedTransformations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/BatchedTransformations.jl/actions/workflows/CI.yml?query=branch%3Amain)

This Julia package provides an interface for applying transformations to data in batches, leveraging [NNlib.jl](https://github.com/FluxML/NNlib.jl) and [Functors.jl](https://github.com/FluxML/Functors.jl) to be GPU-friendly. Moreover, lazy inverse and composition types enable optimization of chain rules.