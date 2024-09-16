using ChainRulesCore
using ChainRulesTestUtils: test_rrule

@testset "ChainRulesCoreExt.jl" begin
    n = 3
    batch_size = (1, 2)
    affine = rand(Float64, Affine, n, batch_size)
    rigid = rand(Float64, Rigid, n, batch_size)

    x = rand(Float64, n, 2, batch_size...)
    test_rrule(transform, affine, x)
    test_rrule(transform, rigid, x)
    test_rrule(inverse_transform, rigid, x)
end