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

	rescale:          if true wil lrescale each column to range [0,1]

	full_compute:     if true will compute the gap statistic for all k 
			      (very slow with increasing N!!), otherwise will return
			      the first k value meeting the criteria

        Returns the optimal cluster number k, and the gap curve if full_compute = true

        ref: https://web.stanford.edu/~hastie/Papers/gap.pdf
    """
    function GapStatistic(X;samples=100, distance=Euclidean(), clustering=kmeans, rescale=true, full_compute=false)
        if rescale
            X = Rescale(X)
        end
        k_max = size(X,1)-1
        S = zeros(size(X,1),size(X,2),samples)
        # generate sample data set (approach (a) in the paper)
        for d in 1:size(X,2)
            S[:,d,:] = rand(Uniform(minimum(X[:,d]),maximum(X[:,d])),(size(X,1),samples))
        end
        gap = zeros(k_max)
        sd = zeros(k_max)
        s = zeros(k_max)
        gapstat = zeros(k_max,2)
            
        Wks = zeros(k_max,samples)
        Wk  = zeros(k_max)
    
        # do k = 1 values
        Wk[1] = Wₖ(X,Int.(ones(size(X,1))),distance)
        for s in 1:samples
            Wks[1,s] = Wₖ(S[:,:,s],Int.(ones(size(X,1))),distance)
        end
    
        l = (1.0/samples)*sum(log.(Wks[1,:]))
        gap[1] = (1.0/samples)*sum(log.(Wks[1,:]).-log(Wk[1]))
        sd[1] = sqrt((1.0/samples)*sum(log.(Wks[1,:]).-l)^2.0)
        s[1] = sd[1]*sqrt(1. + 1.0/samples)

        for k in 2:k_max
             # find the data and synthetic data dispersions for each cluster size
            res = clustering(X',k)
            Wk[k] = Wₖ(X,res.assignments,distance)
            for s in 1:samples
                res = clustering(S[:,:,s]',k)
                Wks[k,s] = Wₖ(S[:,:,s],res.assignments,distance) 
            end
            # compute the gap statistics included 1-std error
            l = (1.0/samples)*sum(log.(Wks[k,:]))
            gap[k] = (1.0/samples)*sum(log.(Wks[k,:]).-log(Wk[k]))
            sd[k] = sqrt((1.0/samples)*sum(log.(Wks[k,:]).-l)^2.0)
            s[k] = sd[k]*sqrt(1. + 1.0/samples)
        
            gapstat[k-1,:] = [gap[k-1],gap[k]-s[k]]
            if full_compute == false && k > 2
                if gapstat[k-2] >= gapstat[k-1]
                    return k-2
                end
            end
        end
        # return the optimal k
    return findfirst(x->x.>=0,gapstat[1:end-1,1].-gapstat[1:end-1,2]),gapstat[1:end-1,1].-gapstat[1:end-1,2]
    end
end # module GapStatistics

