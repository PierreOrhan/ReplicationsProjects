# Here we replicate the model in:
# Simonnet, J., Nassar, M., Stella, F. et al.
#   Activity dependent feedback inhibition may maintain head direction signals in mouse presubiculum.
#   Nat Commun 8, 16032 (2017). https://doi.org/10.1038/ncomms1603
using Distributions
using Flux
using StatsBase
using PyPlot

path = joinpath("src","replicate","replicateSimmonet2017")

## Parameters (Supplementary table 4)
N_pyr = 500
r = 0.25f0
main_w_out = 1f0
w_out = 0.4f0
w_in = 35f0
τ_e = 0.022f0 # 22 ms
τ_i = 0.037f0 # 37 ms τ_i > τ_e --> generating slower input integration times
t_N = 0.005f0 # 5ms :time constant for noise input decay
k1 = 100f0 # parameters for k1
k2 = 0.2f0 # parameter for k2
b1 = 0.00025f0 # controls the persistence of the modulation of synaptic activity <> pyr and MC.
# For increasing b1: faster come back to initial state
b2 = 0.5f0 # controls the direction (b2>0: synapse facilitating; b2<0: synapse in depression)
μ = 0.1f0 # mean of the gaussian used in noise inputs
σ = 0.01f0 #  std of the gaussian used in noise inputs
β = 0.4f0 # strength of the direction selective component of the input on the excitatory cells
κ = 28f0*π/180f0 # degree of selectivity around the central selected diretion of this direction selective component of the input on the excitatory cells
I = 0.6668f0 # ?Lambda?

#Other parameters found in the text:
dt = 0.001f0 # 1 ms
#we fix α = 10, Λ = 1 s,t N_cnn = N_pyr/10
α = 4
Λ = 0.1
N_conn = floor(Int,Λ*N_pyr/α)

#Parameters not found and set-up

θ_bin_e = Float32.(range(0,stop=2π,length=500))
angleDiff(e) = min(abs(e),2π-abs(e))
function get_weight(N_e,N_i)
    #     # inhibitory to pyr:
    N_i = Int(r*N_pyr)
    N_e = N_pyr
    #  "Initially, each Martinotti unit randomly connected to pyramidal units with a 0.7 probability"
    index_connection = rand(Uniform(0,1),(N_e,N_i)).<0.7
    W_out  = Float32.(zeros(N_i,N_e))
    W_in = Float32.(zeros(N_e,N_i))
    # "A sub-set NConn of these inhibitory connections was randomly selected as ‘main connections’ with strength main_w_out"
    #   Later on NComm is given as Λ*N_pyr/α but Λ is not given.
    #   Additionaly we remark that the range α's unit is not clear here, but it is probably not an angle
    choice_main_conn = map(ic->sample(findall(x->x.>0,ic),N_conn,replace=false),eachcol(index_connection))
    #Up: we select a set of main inhibitory connections, below: we we get the indexs of the nearby inhibitory connection to these main connection
    to_prune = map(choice_mart->map(idx->deleteat!(collect(range(idx-Int(α/2),stop=idx+Int(α/2),step=1)),[Int(α/2)+1]),choice_mart),choice_main_conn)
    mod_index(i,max) = i < 1 ? max+i : i>max ? (i-max) : i
    W_in[index_connection] .= 0.01f0
    for j in 1:1:size(to_prune,1) #for each Martinotti cell
        for idx in 1:1:size(choice_main_conn[j],1) #for each main connection (among N_conn)
            # prune connections around a main connection
            W_in[mod_index.(to_prune[j][idx],size(W_in,1)),j] .= 0f0
        end
        #set the gain of the mains connections
        W_out[j,choice_main_conn[j]] .= main_w_out # inhibitory main connections are associated with a reciprocal main excitatory connection
        W_in[choice_main_conn[j],j] .= main_w_out # main connections are set at a high value.

        # among the remaining excitatory connection, we create them with a probability of 0.4
        for i in 1:1:size(W_in,1)
            if W_in[i,j] == 0.01f0
                W_out[j,i] = rand(Uniform(0,1),1)[1] < 0.4 ? rand(Uniform(0f0,w_out),1)[1] : 0f0
            elseif W_in[i,j] == 0f0
                W_out[j,i] = rand(Uniform(0,1),1)[1] < 0.2 ? rand(Uniform(0f0,w_out),1)[1] : 0f0
            end
        end
    end
    # Last step: normalization of the inhibitory connections converging on each pyramidal cell:
    W_in .= W_in ./ sum(W_in,dims=2)
    return W_in,W_out
end
W_in,W_out = get_weight(N_pyr,Int(r*N_pyr))

# Let us proceed to a few analysis of the weights thereby obtained:
fig,ax = plt.subplots(2,2)
Mc_to_pyr = sum(W_in .> 0)/prod(size(W_in))
Pyr_to_mc = sum(W_out .> 0)/prod(size(W_out))
Reciprocal = sum((W_in .>0) .* (transpose(W_out) .> 0))/prod(size(W_in))
ax[1,1].bar(1:1:3,[Mc_to_pyr,Pyr_to_mc,Reciprocal])
ax[1,1].set_xticks(1:1:3)
ax[1,1].set_xticklabels(["Mc→Pyr","Pyr→Mc","Mc↔Pyr"])
ax[1,1].set_ylim(0,1)
#weights plot
ax[2,1].matshow(W_out,aspect="auto")
ax[2,1].xaxis.tick_bottom()
ax[2,1].set_xlabel("ids receiving Pyr")
ax[2,1].set_ylabel("ids emitting  Mcs")
ax[1,2].matshow(W_in,aspect="auto")
ax[1,2].xaxis.set_visible(false)
ax[1,2].yaxis.tick_right()
ax[1,2].set_ylabel("idx emitting Pyr")
ax[2,2].matshow(W_in.*transpose(W_out),aspect="auto")
ax[2,2].set_xlabel("ids  Mcs")
ax[2,2].set_ylabel("idx reciprically coupled Pyr")
ax[2,2].yaxis.tick_right()
display(fig)
fig.savefig(joinpath(path,"figures","weight_distribution.png"))

mutable struct Network
    W_out # connections from pyramidal to Martinotti units
    W_in # connection from Martinotti to pyramidal
    # firing rate
    r_e
    r_i
    # synaptic activity modulation:
    γ_act # for each synapse Pyr -> Martinotti cell (MC)
    γ_eff # for each synapse Pyr -> MC
    #Reciprocal feedback inhibition
    R
end
Network(W_out,W_in,N_e,N_i) = Network(
    W_out,W_in,  #weights
    zeros(Float32,N_e),zeros(Float32,N_i), #rates
    zeros(Float32,(N_i,N_e)),zeros(Float32,(N_i,N_e)),
    zeros(Float32,(N_i,N_e)))

function (N::Network)(h)
    #evolution of synaptic efficacy:
    N.γ_act = N.γ_act .+ b2.*reshape(N.r_e,(1,size(N.r_e,1))) .+ b1*(1f0-sign(b2))/2f0 .- b1*N.γ_act
    N.γ_eff = 1f0./(1f0.+exp.(-k1*(N.γ_act.-k2)))
    # we note that the author used the notation γᴬᶜᵗ_Θ,j(t-1) 🔥 but we understand that they meant t-dt (or equivalently t and for others t+dt)

    # Reciprocal feedback inhibition for each pair is re-computed at every time-step
    N.R = (N.W_out .* reshape(N.r_e,(1,size(N.r_e,1))))
    N.R = N.R ./ sum(N.R,dims=1)
    N.W_in = (1f0.-transpose(N.R)).*N.W_in

    #evolution of the synapses dynamics:
    N.r_i = N.r_i*(1f0-dt/τ_i) .+ relu.((N.γ_eff.*N.W_out)*N.r_e)
    N.r_e = N.r_e*(1f0-dt/τ_e) .+ relu.(h.-N.W_in*N.r_i)

    return N.r_i,N.r_e
end
