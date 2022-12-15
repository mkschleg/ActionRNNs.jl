

import ActionRNNs, Flux

function cell_consistency_test(action_cell)
    Random.seed!(1)
    rnn = action_cell(3, 4, 5)
    ActionRNNs.reset!(rnn, rand(Float32, 5, 16))
    ps = params(rnn)
    grad = Flux.gradient(ps) do
        ret = [rnn(x) for x in [(rand(1:4, 16), rand(Float32, 3, 16)) for _ in 1:5]]
        sum(sum(ret))
    end
    sum(sum.(filter((x)->!isnothing(x), [grad[p] for p in ps])))
end


@testset "Consistency.RingworldER" begin

    

end


