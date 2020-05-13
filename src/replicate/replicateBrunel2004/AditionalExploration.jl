# Additional explorations
using Distributions
using Flux
using LinearAlgebra
using ProgressBars
using PyPlot
using CSV,DataFrames
using Optim
include("currentModels.jl")

function angleDiff(e)
    if abs(e+2.0f0*π) < abs(e-2.0f0*π)
        return abs(e+2.0f0*π) < abs(e) ? abs(e+2.0f0*π)  : abs(e)
    else
        return abs(e-2.0f0*π) < abs(e) ? abs(e-2.0f0*π)  : abs(e)
    end
    return abs(e)
end
function decoder(raster_e,θ_bin_e)
    # Decoder using the population vector (here defined as the vector of size the number of neurons and value  0 or 1, with 1 for a spiking neurons)
    #raster_e: the rasters at each time step
    #θ_bin_e : the prefere angle (rad)
    # here arctan would not directly work. we need to use atan2 (see wikipedia) to obtain the right sign!
    θ_decoded = atan.(sum(sin.(θ_bin_e).*raster_e,dims=1),sum(cos.(θ_bin_e).*raster_e,dims=1)) #output angle in [-π,π]
end

# Question 1: What is the shape of a neuron's tuning curve when the bump is moving at reasonable speed?
α = 50f0*π/180f0
L0 = 0.1f0 #0.02f0
n = Network(1000,1000,α,L0)
T = 15
raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r = simulate(n,T)

θ_bin = -π:2π/60.f0:π-2π/60.f0 #bins of 1 degree
θ = decoder(raster_e,θ_bin_e)[1,:]
raster_binned = map(Ω -> transpose(raster_e)[angleDiff.(θ .- Ω) .< π/60.f0,:],θ_bin)
firing_rates = map(rb->sum(rb,dims=1)/(dt*size(rb,1)),raster_binned)
tc = reduce(vcat,firing_rates)
fig,ax = plt.subplots()
ax.plot(tc[:,503])
display(fig)

# Answer: the shape obtained is as expected gaussian-like with vanishingg tails
# The standard deviation is quite large!


#Question 2: can we generate multiple bumps?
α = 50f0*π/180f0
L0 = 0.1f0 #0.02f0
σ_0 = 4f0
n = Network_noisy_recursive_Weights(1000,1000,α,L0,σ_0)
T = 5
raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r = simulate(n,T)
θ = decoder(raster_e,θ_bin_e)[1,:]
fig,ax = plt.subplots(2,1)
ax[1].plot(θ)
time = range(0,stop=T,length=size(raster_e,2))
events_e = map(r->time[r.==1],eachrow(raster_e))
ax[2].eventplot(events_e)
display(fig)
fig,ax = plt.subplots()
ax.matshow(n.W_lr_gaba)
display(fig)
# Our first explorations with a pure white noise on the recursive weight matrix seems not to be patterning 2 bumps.

# Next step: What happen when the recursive weight matrix also falls of as the distance between inhibitory interneurons?
α = 50f0*π/180f0
α_2 = π
L0 = 0.1f0 #0.02f0
σ_0 = 1f0
n = Network_spatialy_tuned_inhibinhib(1000,1000,α,L0,α_2,σ_0)
T = 5
raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r = simulate(n,T)
θ = decoder(raster_e,θ_bin_e)[1,:]
fig,ax = plt.subplots(4,1)
ax[1].plot(θ)
time = range(0,stop=T,length=size(raster_e,2))
events_e = map(r->time[r.==1],eachrow(raster_e))
events_r = map(r->time[r.==1],eachrow(raster_r))
events_l = map(r->time[r.==1],eachrow(raster_l))
ax[2].eventplot(events_e)
ax[3].eventplot(events_r)
ax[4].eventplot(events_l)
display(fig)
fig,ax = plt.subplots()
ax.matshow(n.W_lr_gaba)
display(fig)
# similar results as previously....


#Next we explore the effect of changing the i->e weights
α = 50f0*π/180f0
L0 = 0.1f0 #0.02f0
n = Network_new_weights(1000,1000,α,L0)
T = 5
raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r = simulate(n,T)
# As we now have 2 bumps moving at the same time, we need to cluster them appart
using Clustering
raster1 = zeros(Float32,size(raster_e))
raster2 = zeros(Float32,size(raster_e))

raster_e = reverse(raster_e,dims=2)
θ_1 = [0.0]
old_center = [0,π] # At each step, we initialize our kmean based on the 2 closes point from the previous center
for tpl in enumerate(eachcol(raster_e))
    idx,r = tpl
    rs = θ_bin_e[r.==1]
    if size(rs,1)>1
        idxMinDold1 = argmin(angleDiff.(rs.-old_center[1]))
        idxMinDold2 = argmin(angleDiff.(rs.-old_center[2]))
        e = kmeans(reshape(rs,(1,size(rs,1))),2,init=[idxMinDold1,idxMinDold2])
        centers = e.centers
        if angleDiff(centers[1]-θ_1[end])<angleDiff(centers[2]-θ_1[end])
            raster1[r.==1,idx] = e.assignments.==1
            raster2[r.==1,idx] = e.assignments.==2
            push!(θ_1,mod(sign(e.centers[1]-θ_1[max(size(θ_1,1)-10,1)])*angleDiff(e.centers[1]-θ_1[max(size(θ_1,1)-10,1)])+θ_1[end]+π,2π)-π)
            old_center[1] = e.centers[1]
            old_center[2] = e.centers[2]
        else
            raster1[r.==1,idx] = e.assignments.==2
            raster2[r.==1,idx] = e.assignments.==1
            push!(θ_1,mod(sign(e.centers[2]-θ_1[max(size(θ_1,1)-10,1)])*angleDiff(e.centers[2]-θ_1[max(size(θ_1,1)-10,1)])+θ_1[end]+π,2π)-π)
            old_center[1] = e.centers[2]
            old_center[2] = e.centers[1]
        end
    elseif size(rs,1)==1
        if angleDiff(rs[1]-θ_1[end])<angleDiff(rs[1]-θ_1[end]+π)
            raster1[r.==1,idx] .= 1
        else
            raster2[r.==1,idx] .=1
        end
        push!(θ_1,θ_1[end])
    else
        push!(θ_1,θ_1[end])
    end
end
events_e1 = map(r->time[r.==1],eachrow(raster1))
events_e2 = map(r->time[r.==1],eachrow(raster2))
fig,ax = plt.subplots()
ax.eventplot(events_e1,color="r")
ax.eventplot(events_e2)
display(fig)
fig,ax =plt.subplots()
ax.scatter(time,θ_1[2:end],s=0.2)
display(fig)

θ = decoder2(raster_e,θ_bin_e)[1,:]
fig,ax = plt.subplots(4,1)
time = range(0,stop=T,length=size(raster_e,2))
ax[1].scatter(time,θ,s=0.001,c="r")
events_e = map(r->time[r.==1],eachrow(raster_e))
events_r = map(r->time[r.==1],eachrow(raster_r))
events_l = map(r->time[r.==1],eachrow(raster_l))
ax1T = ax[1].twinx()
ax1T.eventplot(events_e)
ax[2].eventplot(events_e)
ax[3].eventplot(events_r)
ax[4].eventplot(events_l)
display(fig)
fig,ax = plt.subplots()
ax.matshow(n.W_le_gaba)
display(fig)



θ_bin = -π:2π/120.f0:π-2π/120.f0 #bins of 0.5 degree
θ = decoder2(raster_e,θ_bin_e)[1,:]
raster_binned = map(Ω -> transpose(raster_e)[angleDiff.(θ .- Ω) .< π/60.f0,:],θ_bin)
firing_rates = map(rb->sum(rb,dims=1)/(dt*size(rb,1)),raster_binned)
tc = reduce(vcat,firing_rates)
fig,ax = plt.subplots()
ax.plot(θ_bin,tc[:,1])
display(fig)





# Question 3: Now that we have been able to obtain multiple stable bumps, what is the exact link with the weight matrix?
# Let us analyse their againspectrum:

n = Network_new_weights(1000,1000,α,L0)
fig,ax = plt.subplots()
ax.matshow(n.W_le_gaba)
display(fig)
eigenDecompo = eigen(n.W_le_gaba)
eigVals = eigenDecompo.values
norm_eigVals = norm.(eigVals)
fig,ax = plt.subplots()
ax.plot(norm_eigVals[1:15])
ax.plot(norm_eigVals[end-15:end])
display(fig)

n = Network(1000,1000,α,L0)
fig,ax = plt.subplots()
ax.matshow(n.W_le_gaba)
display(fig)
eigenDecompo = eigen(n.W_le_gaba)
eigVals = eigenDecompo.values
norm_eigVals = norm.(eigVals)
fig,ax = plt.subplots()
ax.plot(norm_eigVals[1:15])
ax.plot(norm_eigVals[end-15:end])
display(fig)


#Exploration: fourrier transform of a sharply tuned excitatory neurons
my_θ = range(0,stop=2π,length=100)
exc_fc(θ) = exp(-100*cos(θ)/(2*σ_e^2))/exp(100/(2*σ_e^2))
fig,ax = plt.subplots()
ax.plot(my_θ,exc_fc.(my_θ))
display(fig)
tc = exc_fc.(my_θ).+0.1*randn(size(my_θ,1)) #let us add a little bit of noise to the tc
g = fft(tc)
fig,ax = plt.subplots(1,2)
ax[1].plot(norm.(g)[1:30])
ax[2].plot(my_θ,tc)
display(fig)

α = 50f0*π/180f0
L0 = 0.1f0 #0.02f0
n = Network(1000,1000,α,L0)
T = 5
raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r = simulate(n,T)
c = cor(raster_e,dims=2)
diagonal(c) .= 0.0
fig,ax = plt.subplots(3,2)
ax[1,1].matshow(c)
ax[1,2].plot(norm.(eigen(c).values))
ax[2,1].matshow(n.W_le_gaba)
ax[2,2].plot(norm.(eigen(n.W_le_gaba).values)[1:30])
ax[3,1].matshow(n.W_re_gaba)
ax[3,2].plot(norm.(eigen(n.W_re_gaba).values)[1:30])
display(fig)

fig,ax = plt.subplots(2,2)
ax[1,1].matshow(n.W_le_gaba.+n.W_re_gaba)
ax[1,2].matshow(n.W_le_gaba)
ax[2,2].matshow(n.W_re_gaba)
ax[2,1].matshow(c)
display(fig)
