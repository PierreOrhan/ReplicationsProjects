using PyPlot
using PyCall
using FFTW
include("layers.jl")


mutable struct boardGrid
    l::OneDstdpLayer
    mem_activ_e #save activations of trajectory
    mem_activ_i
    save_si # initial activations for reset for learning
    save_se
    tc_i #tuning curves
    tc_e
    gs_i # grid scores
    gs_e
    pop_gs_i #population grid scores
    pop_gs_e
    bin
end
boardGrid(l::OneDstdpLayer,usingGPU::Bool) =
    boardGrid(l,
    [], #save activations of trajectory
    [],
    usingGPU ? gpu(zeros(Float32,size(l.s_i))) : zeros(Float32,size(l.s_i)) , # initial activations for reset for learning
    usingGPU ? gpu(zeros(Float32,size(l.s_e))) : zeros(Float32,size(l.s_e)),
    [], #tuning curves
    [],
    [], # grid scores
    [],
    [], #population grid score
    [],
    0:0.01f0:1)
function prepareBoard(b::boardGrid)
    b.save_si[:] = b.l.s_i
    b.save_se[:] = b.l.s_e
    reset_synapse(b.l)
    b.mem_activ_e = []
    b.mem_activ_i = []
end
function (b::boardGrid)(inputs_e,inputs_i,xshunt_e,xshunt_i)
    se,si = sampling(b.l,inputs_e,inputs_i,xshunt_e,xshunt_i)
    push!(b.mem_activ_i,cpu(si))
    push!(b.mem_activ_e,cpu(se))
end
function boxCarFilter(X::Array{Float32,1},width::Int64)
    Xsmooth = zeros(Float32,size(X))
    for idx in 1:1:width
        Xsmooth[idx] = sum(X[1:idx])/idx
    end
    for idx in width+1:1:size(X,1)
        Xsmooth[idx] = Xsmooth[idx-1] + (X[idx] - X[idx-width])/width
    end
    return Xsmooth
end
function boxCarFilter(X,width::Int64)
    Xsmooth = zeros(Float32,size(X))
    for idx in 1:1:width
        Xsmooth[:,idx] = sum(X[:,1:idx],dims=2)/idx
    end
    for idx in width+1:1:size(X,2)
        Xsmooth[:,idx] = Xsmooth[:,idx-1] .+ (X[:,idx] .- X[:,idx-width])/width
    end
    return Xsmooth
end
function computeTC(b::boardGrid,X_input)
    # compute the tuning curves given the saved activity
    X_bin = b.bin
    # bins of 5 cm --> procedure could be improved by computing bin_size that maximizes mutual information
    if b.l.isStochasticDynamics
        raster_train = transpose(reduce(hcat,b.mem_activ_e)) #(tstep,nbNeurons)
        raster_binned = map(x -> raster_train[abs.(X_input .- x) .< 0.025f0,:],X_bin)
        firing_rates = map(rb->sum(rb,dims=1)/(dt*size(rb,1)),raster_binned)
        tc = transpose(reduce(vcat,firing_rates))
        #the tuning curve is then smoothed using a boxcar filter of width = 5 bins:
        tc = boxCarFilter(tc,5)
        push!(b.tc_e,tc)
        push!(b.gs_e,grid_score(tc))
        #For the population activity:
        # We decompose the test trajectory (size(X_bin,1)*dt seconds) into L blocks of 5 seconds
        # The plot will focus on the average and standard deviations of the gridness of these snapshot of activity vector
        # gridness is here seen through the lens of the population activity vector
        # In addition we use only half of the number of neurons....

        Ldt = Int64(div(size(raster_train,1),5))
        subdivided_raster_train = map(idx->raster_train[idx:idx+Ldt,1:div(size(raster_train,2),2)],1:Ldt:size(raster_train,1)-Ldt)
        pop_activity = transpose(reduce(vcat,mean.(subdivided_raster_train,dims=1)))
        push!(b.pop_gs_e,grid_score(pop_activity))

        #Same for inhibitory
        raster_train = transpose(reduce(hcat,b.mem_activ_i)) #(tstep,nbNeurons)
        raster_binned = map(x -> raster_train[abs.(X_input .- x) .< 0.025f0,:],X_bin)
        firing_rates = map(rb->sum(rb,dims=1)/(dt*size(rb,1)),raster_binned)
        tc = transpose(reduce(vcat,firing_rates))
        tc = boxCarFilter(tc,5)
        push!(b.tc_i,tc)
        push!(b.gs_i,grid_score(tc))

        Ldt = Int64(div(size(raster_train,1),5))
        subdivided_raster_train = map(idx->raster_train[idx:idx+Ldt,1:div(size(raster_train,2),2)],1:Ldt:size(raster_train,1)-Ldt)
        pop_activity = transpose(reduce(vcat,mean.(subdivided_raster_train,dims=1)))
        push!(b.pop_gs_i,grid_score(pop_activity))
    else
        ρ_train = transpose(reduce(hcat,b.mem_activ_e))
        ρ_binned = map(x -> ρ_train[abs.(X_input .- x) .< 0.025f0,:],X_bin)
        firing_rates = map(rb->sum(rb,dims=1)/(size(rb,1)),ρ_binned)
        tc = reduce(vcat,firing_rates)'
        tc = boxCarFilter(tc,5)
        push!(b.tc_e,tc)
        push!(b.gs_e,grid_score(tc))
        Ldt = Int64(div(size(ρ_train,1),5))
        subdivided_raster_train = map(idx->ρ_train[idx:idx+Ldt,1:div(size(ρ_train,2),2)],1:Ldt:size(ρ_train,1)-Ldt)
        pop_activity = transpose(reduce(vcat,mean.(subdivided_raster_train,dims=1)))
        push!(b.pop_gs_e,grid_score(pop_activity))

        ρ_train = transpose(reduce(hcat,b.mem_activ_i))
        ρ_binned = map(x -> ρ_train[abs.(X_input .- x) .< 0.025f0,:],X_bin)
        firing_rates = map(rb->sum(rb,dims=1)/(size(rb,1)),ρ_binned)
        tc = transpose(reduce(vcat,firing_rates))
        tc = boxCarFilter(tc,5)
        push!(b.tc_i,tc)
        push!(b.gs_i,grid_score(tc))
        subdivided_raster_train = map(idx->ρ_train[idx:idx+Ldt,1:div(size(ρ_train,2),2)],1:Ldt:size(ρ_train,1)-Ldt)
        pop_activity = transpose(reduce(vcat,mean.(subdivided_raster_train,dims=1)))
        push!(b.pop_gs_i,grid_score(pop_activity))
    end
end
function myplot(b::boardGrid)
    Mat_e = reduce(hcat,b.gs_e)
    Mat_i = reduce(hcat,b.gs_i)

    axes_grid1 = pyimport("mpl_toolkits.axes_grid1")
    fig,ax = plt.subplots(1,2)
    #prepare colormap
    divider1 = axes_grid1.make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="10%", pad=0.05)
    divider2 = axes_grid1.make_axes_locatable(ax[2])
    cax2 = divider2.append_axes("right", size="10%", pad=0.05)
    cmap="cool"
    im1 = ax[1].imshow(Mat_e,cmap=cmap)
    im2 = ax[2].imshow(Mat_i,cmap=cmap)

    #normalize the colormap between 0 and 1
    vmin = 0
    vmax = 1
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    im1.set_norm(norm)
    im2.set_norm(norm)

    fig.colorbar(im2, cax=cax1, orientation="vertical")
    fig.colorbar(im2, cax=cax2, orientation="vertical")
    display(fig)

    mean_e = mean(Mat_e,dims=1)[1,:]
    mean_i = mean(Mat_i,dims=1)[1,:]
    std_e = std(Mat_e,dims=1)[1,:]
    std_i = std(Mat_i,dims=1)[1,:]
    fig,ax = plt.subplots(1,2)
    ax[1].plot(mean_e,label="excitatory tc gridness",c="b")
    ax[1].fill_between(0:1:size(mean_e,1)-1,mean_e .+ std_e,mean_e .-std_e,alpha=0.5f0)
    ax[1].plot(mean_i,label="inhibitory tc gridness",c="r")
    ax[1].fill_between(0:1:size(mean_i,1)-1,mean_i .+ std_i,mean_i .-std_i,alpha=0.5f0)
    ax[1].set_ylim(0f0,1f0)

    Mat_e_pop  = reduce(hcat,b.pop_gs_e)
    Mat_i_pop  = reduce(hcat,b.pop_gs_i)
    mean_e_pop = mean(Mat_e_pop,dims=1)[1,:]
    mean_i_pop = mean(Mat_i_pop,dims=1)[1,:]
    std_e_pop = std(Mat_e_pop,dims=1)[1,:]
    std_i_pop = std(Mat_i_pop,dims=1)[1,:]
    print(mean_e_pop)
    print(std_e_pop)
    ax[2].plot(mean_e_pop,label="excitatory pop vector gridness",c="g")
    ax[2].fill_between(0:1:size(mean_e_pop,1)-1,mean_e_pop .+ std_e_pop,mean_e_pop .-std_e_pop,alpha=0.5f0)
    ax[2].plot(mean_i_pop,label="inhibitory pop vector gridness",c="m")
    ax[2].fill_between(0:1:size(mean_i_pop,1)-1,mean_i_pop .+ std_i_pop,mean_i_pop .-std_i_pop,alpha=0.5f0)
    # ax[2].set_ylim(0f0,1f0)
    fig.legend()
    display(fig)
end
function plot_tc(b::boardGrid)
    plot_learnedHD_Fields(b.tc_e[end],b.bin,"excitatory cells")
    plot_learnedHD_Fields(b.tc_i[end],b.bin,"inhibitory cells")
end


#utils: grid score computation
# giving a firing rate
#   we define the grid score as the power of the higghest frequency of the power spectrum
#       assuming the grid pattern would repeat symetrcically and then be perioric.
using FFTW
function modIndex(idx,max)
    idx > max ? modIndex(idx-max,max) : idx < 1 ? modIndex(max+idx,max) : idx
end
function periodicautoCor(x) #
    return u -> sum(map(idx->x[idx]*x[modIndex(idx+u,size(x,1))],1:1:size(x,1)))
end
function autoCC(X)
    # X: (neurons,firing_rates_bin over 2meters, 0:1meters + 1:0)
    reduce(hcat,map(idx -> periodicautoCor(X[idx,:]).(1:1:(size(X,2))),1:1:size(X,1)))
end
function grid_score(tc)
    # tc: (neurons,firing_rates_bin over 0:1 meters)
    tc = tc.-mean(tc,dims=2)
    autocc = autoCC(tc)
    fftAcc = reduce(hcat,fft.(eachcol(autocc)))[1:div(size(autocc,1),2),:]
    power = transpose(norm.(fftAcc))
    powerProba = transpose(reduce(hcat,map(p->sum(p) ==0 ? zeros(size(p,1)) : p / sum(p),eachrow(power))))
    pmax = maximum.(eachrow(powerProba))
    return pmax
end

function intToGrid(x::Int)
    square = sqrt(x)
    floorSquare = Int(floor(square))
    if floorSquare==square
        return (floorSquare,floorSquare)
    elseif (floorSquare+1)*floorSquare>=x
        return (floorSquare+1,floorSquare)
    else
        return (floorSquare+1,floorSquare+1)
    end
end

function plot_learnedHD_Fields(activations,binTrack,figTitle;colorType=nothing)
    sizeGrid = intToGrid(size(activations,1))
    fig = plt.figure(figsize = sizeGrid .* 10)
    posInGrid = reduce(vcat,collect(Iterators.product(1:sizeGrid[1],1:sizeGrid[2])))
    if posInGrid == (1,1)
        posInGrid = [(1,1)]
    end
    c= plt.get_cmap("tab10")
    for idx in 1:1:size(activations,1)
        ax = plt.subplot2grid(sizeGrid,(posInGrid[idx][1]-1,posInGrid[idx][2]-1))
        if colorType == nothing
            ax.plot(binTrack,activations[idx,:])
        else
            ax.plot(binTrack,activations[idx,:],c=c(colorType[idx]))
        end
    end
    fig.suptitle(string("Activations","fields",string(figTitle)),fontsize=sizeGrid[1]*10)
    plt.tight_layout()
    display(fig)
    # fig.savefig(joinpath(path,"Activations","fields",string(figTitle,".png")))
    # plt.close(fig)
end
