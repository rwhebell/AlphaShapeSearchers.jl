module AlphaShapeSearchers

using LinearAlgebra, StaticArrays, Delaunay, Random, Distances, StatsBase, AndExport

@xport struct AlphaShapeSearcher{CoordType, dim}

    X::Matrix{CoordType}
    simplices::Matrix{Int}
    circumcenters::Matrix{CoordType}
    radii::Vector{CoordType}
    α::CoordType
    edge2simplex::Dict{Vector{Int}, Vector{Int}}

end # struct AlphaShapeSearcher

@xport function AlphaShapeSearcher(X::Matrix{T}, α=Inf) where {T<:Real}

    d, nx = size(X)

    triangulation = X |> transpose |> collect |> delaunay
    simplices = triangulation.simplices |> transpose |> collect
    triangulation = nothing

    k, ns = size(simplices)

    radii = zeros(T, ns)
    circumcenters = zeros(T, d, ns)

    for (i,s) in enumerate(eachcol(simplices))
        Xs = X[:,s]
        centroid = mean(Xs, dims=2)
        Xs = Xs .- centroid
        circumcenters[:,i], radii[i] = circumsphere(Xs)
        circumcenters[:,i] .+= centroid
    end

    edge2simplex = buildIncidence(simplices)

    return AlphaShapeSearcher{T, d}(X, simplices, circumcenters, radii, α, edge2simplex)

end

function AlphaShapeSearcher(αshape::AlphaShapeSearcher{T,d}, α) where {T,d}
    return AlphaShapeSearcher{T, d}(αshape.X, αshape.simplices, αshape.circumcenters, 
        αshape.radii, α, αshape.edge2simplex)
end

function circumsphere(V::Matrix{S}) where S
    s, nv = size(V)
    A = [ 2*transpose(V)*V  ones(S,nv,1);  ones(S,1,nv)  zero(S) ]
    b = [ [ norm(V[:,i])^2 for i in 1:nv ]; one(S) ]
    alpha = (A\b)[1:nv]
    c = V*alpha
    r = norm(V[:,1] - c)
    return c, r
end

function pickStartingEdge(X, S, q)

    d, nx = size(X)
    k, ns = size(S)

    m = ceil(Int, nx^(1/4))

    sampleIDs = sample(1:ns, m, replace=false)

    _, closestInd = findmin(sampleIDs) do i
        minimum(1:k) do j
            SqEuclidean()(@view(X[:, S[j,i]]), q)
        end
    end

    closest = sampleIDs[closestInd]
    
    # Starting edge
    edge = S[1:k-1, closest] |> sort
    
    return edge
    
end

@xport function inshape(αshape::AlphaShapeSearcher{T,d}, q) where {T,d}
    id = inTriangulation(αshape.X, αshape.simplices, αshape.edge2simplex, 
        αshape.circumcenters, αshape.radii, q, Val(d))
    return (id > 0 && αshape.radii[id] ≤ αshape.α)
end

function inTriangulation(X::Matrix{T}, S, edge2simplex, C, R, q, ::Val{dim}) where {T, dim}
    # Based on:
    # Ernst P. Mücke, Isaac Saias, Binhai Zhu.
    # "Fast randomized point location without preprocessing in two- and three-dimensional Delaunay triangulations"
    # Computational Geometry 12 (1999) 63–83
    # https://doi.org/10.1016/S0925-7721(98)00035-2
    
    k = dim+1 # number of points in a simplex in R^dim

    edge = pickStartingEdge(X, S, q)
    new_edge = zero(edge)
    inds = zeros(Int, k-1)

    id = edge2simplex[edge][1]
    simplex = S[:,id]
    oppPoint = getOppPoint(simplex, edge)

    edge_indices = collect(1:k)

    # TODO: abolish this storage of n
    # Could instead reverse the order of P's columns, or better still, reverse `edge`
    P = MMatrix{dim,dim,T}(X[:, edge])
    n = MVector{dim,T}(X[:,oppPoint])

    Pq = zero(P)
    Pn = zero(P)

    A = @MMatrix zeros(T, dim+1, dim+1)
    b = @MVector zeros(T, dim+1)

    # ensure we start with q on the positive side of the edge
    if length(edge2simplex[edge]) > 1
        if !different_side_test!(Pq, Pn, P, n, q)
            id = edge2simplex[edge][2]
            simplex .= S[:,id]
            oppPoint = getOppPoint(simplex, edge)
            n .= X[:, oppPoint]
        end
    end

    it = 0

    while it < size(S,2)

        it += 1

        if length(edge2simplex[edge]) == 1 # boundary edge

            oppPoint = getOppPoint(simplex, edge)
            P .= X[:, edge]
            n .= X[:, oppPoint]

            if different_side_test!(Pq, Pn, P, n, q)
                return 0
            end

        end

        # Is q in this simplex?
        if Euclidean(1e-12)(C[:,id], q) ≤ R[id] && inSimplex!( A, b, @view(X[:, simplex]), q )
            return id
        end

        # If not, pick a new edge of the other simplex incident to `edge'
        if id == edge2simplex[edge][1] && length(edge2simplex[edge]) > 1
            id = edge2simplex[edge][2]
        else
            id = edge2simplex[edge][1]
        end

        # simplex .= S[:,id]
        copyto!(simplex, 1, S, k*(id-1)+1, k)

        for f in shuffle!(edge_indices) # for each edge of the new simplex

            inds[1:f-1] .= 1:f-1
            inds[f:end] .= f+1:k
            new_edge .= @view S[inds, id]
            sort!(new_edge)
            if new_edge == edge # not the old edge
                continue
            end

            new_oppPoint = getOppPoint(simplex, new_edge)
            P .= @view X[:, new_edge]
            n .= @view X[:, new_oppPoint]
            
            if different_side_test!(Pq, Pn, P, n, q) # this has to be true for one of the edges
                edge .= new_edge
                oppPoint = new_oppPoint
                break
            end

        end

        # now we have a new simplex and a new edge, with q on the positive side

    end

    return -1 # in case we had an infinite loop or something (rare but possible in case of numerical weirdness)

end

function getOppPoint(elem, edge)
    return elem[ findfirst(!in(edge), elem) ]
end

function buildIncidence(S::Matrix{Int})
    d, nt = size(S)
    incidence = Dict{Vector{Int}, Vector{Int}}()
    for i in 1:d
        inds = [1:i-1; i+1:d]
        for t in 1:nt
            edge = S[inds,t] |> sort!
            if edge ∈ keys(incidence)
                push!(incidence[edge], t)
            else
                incidence[edge] = [t]
            end
        end
    end
    return incidence
end

function different_side_test!(Pq, Pn, P, n, q)

    # P has d-1 points in R^d as its columns.
    # (these uniquely define a hyperplane)
    # n is a point on the negative side of the plane.
    # q is the query point.
    # Is q on the positive side of the plane?

    d, m = size(P)
    
    # In dimension d, the face has d points.
    @assert d == m
    @assert size(Pq) == size(Pn) == (d,m)

    leaveOut = 1
    smallDetWarning = true
    detq = 0.0
    detn = 0.0

    while smallDetWarning && leaveOut ≤ m

        @inbounds begin

            copyto!(Pq, CartesianIndices((1:d, 1:leaveOut-1)), P, CartesianIndices((1:d, 1:leaveOut-1)))
            copyto!(Pn, CartesianIndices((1:d, 1:leaveOut-1)), P, CartesianIndices((1:d, 1:leaveOut-1)))

            copyto!(Pq, CartesianIndices((1:d, leaveOut:m-1)), P, CartesianIndices((1:d, leaveOut+1:m)))
            copyto!(Pn, CartesianIndices((1:d, leaveOut:m-1)), P, CartesianIndices((1:d, leaveOut+1:m)))
            
            Pq[:,m] .= q
            Pn[:,m] .= n

            for i in 1:m
                Pq[:,i] -= P[:,leaveOut]
                Pn[:,i] -= P[:,leaveOut]
            end

        end

        detq = det(Pq)
        detn = det(Pn)

        smallDetWarning = (abs(detq) < eps()) || (abs(detn) < eps())

        leaveOut += 1

    end

    q_sign = sign(detq)
    n_sign = sign(detn)

    # @assert n_sign != 0 "P=$P; q=$q; n=$n"

    return q_sign != n_sign

    # determinant-based check:
    # https://math.stackexchange.com/questions/2214825/determinant-connection-to-deciding-side-of-plane

end

function inSimplex!(A, b, P, q)

    d, n = size(P)
    @assert n == d+1
    @assert size(A,1) == d+1
    @assert size(A,2) == n
    @assert size(b) == size(q).+1

    j = rand(1:n)

    A[1:end-1, :] .= P .- P[:,j]
    A[end, :] .= 1

    b[1:end-1] .= q .- P[:,j]
    b[end] = 1

    A_fac = lu!(A, check=false)

    if !issuccess(A_fac)
        A[1:end-1, :] .= P .- P[:,j]
        A[end, :] .= 1
        A_fac = qr!(A, ColumnNorm())
    end

    lambda = ldiv!(A_fac, b)
    return all(>(-eps()), lambda)

end

end # module AlphaShapeSearchers
