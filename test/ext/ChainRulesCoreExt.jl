using ChainRulesCore
using ChainRulesTestUtils: test_rrule

@testset "ChainRulesCoreExt.jl" begin
    n = 3
    batch_size = (1, 2)
    rigid = rand(RigidTransformations, Float64, n, batch_size)

    x = rand(Float64, n, 2, batch_size...)
    test_rrule(transform, rigid, x)
    test_rrule(inverse_transform, rigid, x)
end