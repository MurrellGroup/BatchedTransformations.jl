using BatchedTransformations
using Documenter

DocMeta.setdocmeta!(BatchedTransformations, :DocTestSetup, :(using BatchedTransformations); recursive=true)

makedocs(;
    modules=[BatchedTransformations],
    authors="anton083 <anton.oresten42@gmail.com> and contributors",
    sitename="BatchedTransformations.jl",
    format=Documenter.HTML(;
        canonical="https://anton083.github.io/BatchedTransformations.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/anton083/BatchedTransformations.jl",
    devbranch="main",
)
