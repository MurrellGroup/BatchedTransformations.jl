using BatchedTransformations
using Documenter

DocMeta.setdocmeta!(BatchedTransformations, :DocTestSetup, :(using BatchedTransformations); recursive=true)

makedocs(;
    modules=[BatchedTransformations],
    authors="anton083 <anton.oresten42@gmail.com> and contributors",
    sitename="BatchedTransformations.jl",
    doctest=true,
    format=Documenter.HTML(;
        canonical="https://murrellgroup.github.io/BatchedTransformations.jl",
        edit_link="main",
        assets=String[],
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/BatchedTransformations.jl.git",
    devbranch="main",
)
