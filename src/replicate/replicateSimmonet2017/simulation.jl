# Here we replicate the model in:
# Simonnet, J., Nassar, M., Stella, F. et al.
#   Activity dependent feedback inhibition may maintain head direction signals in mouse presubiculum.
#   Nat Commun 8, 16032 (2017). https://doi.org/10.1038/ncomms1603

using Distributions
using Flux
using StatsBase
using PyPlot
using ProgressBars
include("model.jl")
path = joinpath("src","replicate","replicateSimmonet2017")

NbSteps = 1000
#input noise
h_noise = zeros(Float32,size(θ_bin_e))
Random_Noise = rand(Normal(μ,σ),(size(θ_bin_e,1),NbSteps))
#simulation 1: white noise (correlated in time)

n = Network(W_out,W_in,N_pyr,Int(r*N_pyr))
#initialize the excitatory firing rate:
n.r_e .= n.r_e .+ 0.01f0
rates_e,rates_i = [],[]
for tstep in ProgressBar(1:1:NbSteps)
    h_noise .= h_noise*(1f0-dt/t_N) .+ Random_Noise[:,tstep]
    ri,re = n(h_noise)
    push!(rates_e,re)
    push!(rates_i,ri)
end
red_rates_e =  reduce(hcat,rates_e)
red_rates_i = reduce(hcat,rates_i)
times = 1:1:size(red_rates_e,2)
events_e = map(r->times[r.>0.01f0],eachrow(red_rates_e))
fig,ax = plt.subplots(2,1)
ax[1].eventplot(events_e)
ax[2].plot(mean(red_rates_e,dims=1)[1,:])
display(fig)



# simulation 2: white noise (correlated in time) + slective inputs
h_selective = zeros(Float32,(size(θ_bin_e,1),size(θ_bin_e,1)*200))
decay_factor = reshape(exp.(-(1f0:1f0:20f0)/10f0),(1,20))
for idx in 0:1:size(θ_bin_e,1)-1
    #activation of the new angle
    activation = reshape(exp.(-angleDiff.(θ_bin_e.-θ_bin_e[idx+1]).^2/(2f0*κ^2)),(size(θ_bin_e,1),1))
    h_selective[:,idx*200+1:idx*200+60] .= β* activation.*ones(size(θ_bin_e,1),60)
    #slow decay:
    h_selective[:,idx*200+61:idx*200+80] .= β*activation .* (decay_factor.*ones(size(θ_bin_e,1),20))
    # let the network evolve for the remaining 178 ms:
    h_selective[:,idx*200+81:(idx+1)*200] .= 0
end
n = Network(W_out,W_in,N_pyr,Int(r*N_pyr))
#input noise
h_noise = zeros(Float32,size(θ_bin_e))
Random_Noise = rand(Normal(μ,σ),(size(θ_bin_e,1),size(θ_bin_e,1)*200))
rates_e,rates_i = [],[]
for tstep in ProgressBar(1:1:size(h_selective,2))
    h_noise .= h_noise*(1f0-dt/t_N) .+ Random_Noise[:,tstep]
    ri,re = n(h_noise.+h_selective[:,tstep])
    push!(rates_e,re)
    push!(rates_i,ri)
end

red_rates_e =  reduce(hcat,rates_e)
red_rates_i = reduce(hcat,rates_i)

times = 1:1:size(red_rates_e,2)
events_e = map(r->times[r.>0.01f0],eachrow(red_rates_e))
fig,ax = plt.subplots()
ax.eventplot(events_e)
display(fig)

#Reproducing supp figure 5:
# we can deduce the tuning curve for this experiment:
tc_e = zeros(Float32,(size(θ_bin_e,1),N_pyr))
tc_i = zeros(Float32,(size(θ_bin_e,1),Int(r*N_pyr)))
for idx in 0:1:size(θ_bin_e,1)-1
    tc_e[idx+1,:] .= mean(red_rates_e[:,idx*200+1:(idx+1)*200],dims=2)[:,1]
    tc_i[idx+1,:] .= mean(red_rates_i[:,idx*200+1:(idx+1)*200],dims=2)[:,1]
end

choice = [250,180,499,140,400]
fig = plt.figure()
for idx in 1:1:size(choice,1)
    ax = fig.add_subplot(2,5,idx,projection = "polar")
    ax.plot(θ_bin_e,tc_e[:,idx],c="b")
    ax = fig.add_subplot(2,5,5+idx,projection = "polar")
    ax.plot(θ_bin_e,tc_i[:,idx],c="b")
end
display(fig)
