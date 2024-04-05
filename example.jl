using LinearAlgebra, StatsBase, AlphaShapeSearchers

dim = 3
N = 10_000

X = rand(dim, N)

αshape = AlphaShapeSearcher(X, Inf) # the convex hull

# test for a random convex combination of 10 points in X
J = sample(1:N, 10)
q = mapreduce(*, +, normalize!(rand(10),1), eachcol(X[:,J])) 
flag = inshape(αshape, q)
@assert flag
