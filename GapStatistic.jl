module GapStatistics

    import Distributions.Uniform
    using Distances
    using Clustering
    export GapStatistic

    function Rescale(x)
        y = copy(x)
        for d in 1:size(x,2)
            y[:,d] = (x[:,d] .- minimum(x[:,d]))/(maximum(x[:,d])-minimum(x[:,d]))
        end
        return y
    end

    Dᵣ(X,distance) = sum(pairwise(distance,X'))::Float64

    """
        X:                Nxd set of d-dimensional points

        assignments:      N assignment indices

        distance:         Distsances.jl distance function

        Returns the dispersion of the point set

        ref: https://web.stanford.edu/~hastie/Papers/gap.pdf
    """
    function Wₖ(X,assignments,distance)::Float64
        w = 0.0
        for k in unique(assignments)
            a = findall(x->x.==k,assignments)
            x = X[a,:]
            w += (1.0/(2.0*length(a)))*Dᵣ(x,distance)
        end
        return w
    end

    """
        X:                Nxd set of d-dimensional points

        clustering:       Clustering.jl clustering algorithm (actually any function that takes
                              the points and a cluster number k and returns 
                              clustering assignments as an N dimensional Array{Int} accessed like
                              res = clustering(X,k); res.assignments)

        distance:         Distsances.jl distance function

        Returns the optimal cluster number k and the gap curve

        ref: https://web.stanford.edu/~hastie/Papers/gap.pdf
    """
    function GapStatistic(X;samples=100, distance=Euclidean(), clustering=kmeans, rescale=true)::Tuple{Int,Array{Float64}}
        if rescale
            X = Rescale(X)
        end
        k_max = size(X,1)-1
        S = zeros(size(X,1),size(X,2),samples)
        # generate sample data set (approach (a) in the paper)
        for d in 1:size(X,2)
            S[:,d,:] = rand(Uniform(minimum(X[:,d]),maximum(X[:,d])),(size(X,1),samples))
        end
        # find the data and synthetic data dispersions for each cluster size
        Wks = zeros(k_max,samples)
        Wk  = zeros(k_max)
        for k in 2:k_max
            res = clustering(X',k)
            Wk[k] = Wₖ(X,res.assignments,distance)
            for s in 1:samples
                res = clustering(S[:,:,s]',k)
                Wks[k,s] = Wₖ(S[:,:,s],res.assignments,distance) 
            end
        end

        # do k = 1
        Wk[1] = Wₖ(X,Int.(ones(size(X,1))),distance)
        for s in 1:samples
            Wks[1,s] = Wₖ(S[:,:,s],Int.(ones(size(X,1))),distance)
        end

        # compute the gap statistics included 1-std error
        gap = zeros(k_max)
        sd = zeros(k_max)
        s = zeros(k_max)
        for k in 1:k_max
            l = (1.0/samples)*sum(log.(Wks[k,:]))
            gap[k] = (1.0/samples)*sum(log.(Wks[k,:]).-log(Wk[k]))
            sd[k] = sqrt((1.0/samples)*sum(log.(Wks[k,:]).-l)^2.0)
            s[k] = sd[k]*sqrt(1. + 1.0/samples)
        end
        gapstat = zeros(k_max,2)
        for k in 1:k_max-1
            gapstat[k,:] = [gap[k],gap[k+1]]
        end
        # return the optimal k
    return findfirst(x->x.>=0,gapstat[1:end-1,1].-gapstat[1:end-1,2]),gapstat[1:end-1,1].-gapstat[1:end-1,2]
    end
end # module GapStatistics

using Main.GapStatistics
