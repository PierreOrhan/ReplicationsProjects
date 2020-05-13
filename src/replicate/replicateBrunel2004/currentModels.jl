using Distributions
using Flux
using LinearAlgebra
using ProgressBars
using PyPlot

#🔥 Model as described in Brunel2004 paper.
# This model uses ampa, nmda and gaba receptor. We use here an integration with the Euler strategy and time step of 0.5ms.
# To capture the dynamic a bit better, one can use a time-step of 0.02 ms, as described in the XJ-Wang paper.
# No improvement is made in term of computation time when used on the GPU, as not matrix multiplication is used in this implementation.

#Parameters
Ce = 0.5f0 #0.5 nG
ge = 25f0 #25 nS
τe = 0.02f0 #20ms 🔥
Cl = 0.2f0
Cr = 0.2f0
gl = 20f0
gr = 20f0
τl = 0.01f0 #10 ms 🔥
τr = 0.01f0
g_gaba = 1.25f0 #1.25 nS
V_I = -70f0 #-70mV
g_ampa = 0.016f0 # 0.016nS
V_E = 0f0 #0mV
g_NMDA = 0.258f0 #00.258 nS
gE_ampa = 2.08f0 # 2.08 nS
gI_ampa = 1.62f0 # 1.62 nS
Mg2 = 1f0 #[Mg²⁺] = 1mM
τ_decay_ampa = 0.002f0 # 2ms
τ_decay_gaba = 0.005f0 # 5ms
τ_decay_nmda = 0.1f0 # 100ms
τ_ref_e = 0.002f0 #2ms
τ_ref_l = 0.001f0 # 1ms
τ_ref_r = 0.001f0 # 1ms
V_leak = -70f0 #-70mV
V_treshold = -50f0 # -50mV
V_reset = -55f0 # -55mV

v_to_I = 5.5f0 #conversion from speed input into current: not provided in the paper. We obtained it manually with a few trials, the goal is just to reproduce the shape of the figures for figure 9
ext_strength_i =  1f0#3.5f0/1.62f0 # strength factor for the synaptic connection between background inputs and neurons in the network
ext_strength_e = 1f0#5.07f0/2.08f0
#simulation parameters
dt = 0.0005f0 # 0.5 ms

#weight parameters
function angleDiff(e)
    if abs(e+2.0f0*π) < abs(e-2.0f0*π)
        return abs(e+2.0f0*π) < abs(e) ? abs(e+2.0f0*π)  : abs(e)
    else
        return abs(e-2.0f0*π) < abs(e) ? abs(e-2.0f0*π)  : abs(e)
    end
    return abs(e)
end

K1 = 1.1f0
# K1 = 0f0
σ_l = 30f0*π/180f0
K_dtn_lmn(θ) = K1*exp(-(angleDiff(θ)/σ_l)^2/2)
H1 = 0.065f0
# H1 =0f0
σ_e = 80f0*π/180f0
H_lmn_dtn(θ) = H1*exp(-(-angleDiff(θ)/σ_e)^2/2)

function weight_init(N1::Int,N2::Int,phase,f)
    w = zeros(Float32,(N1,N2))
    θ_bin_1 = range(-π,stop=π,length=N1)
    θ_bin_2 = range(-π,stop=π,length=N2)
    for i in 1:1:N1
        for j in 1:1:N2
            w[i,j] = f(-θ_bin_1[i]+θ_bin_2[j]+phase)
        end
    end
    w
end
function weight_init(N1::Int,N2::Int,L0::Float32)
    w = zeros(Float32,(N1,N2)) .+ L0
end
anelu(x) = relu(-1f0 .* x)

mutable struct Network
    N_l #Number of inhibitory in LMN-l
    N_r #Number of inhibitory in LMN-r
    N_e #Number of excitatory in DTN
    V_l #membrane potentials
    V_r
    V_e
    #frozen memories: counters that starts at -div(τ_ref,dt) and are increased by one until they reached the value 1 where they are not increased anymore.
    # The Neuron dynamic then starts again.
    frozen_l
    frozen_r
    frozen_e
    #activation of synapes for LMN-l -> A
    s_le_gaba # 🔥 these variables are vector where each elements is the sum of activations from synapses of pop l onto pop e 🔥
    s_lr_gaba
    #activation of synapes for LMN-r -> A
    s_re_gaba
    s_rl_gaba
    #activation of synapes for DTN-e -> A
    s_el_nmda
    s_er_nmda
    s_el_ampa
    s_er_ampa
    #external inputs: AMPA synaptic receptors (For Ii and IE)
    s_ext_ampa_l
    s_ext_ampa_r
    s_ext_ampa_e

    #Weights
    #Weights of synapes for LMN-l -> A (A=e or r)
    W_le_gaba
    W_lr_gaba
    #Weights of synapes for LMN-r -> A
    W_re_gaba
    W_rl_gaba
    #Weights of synapes for DTN-e -> A
    W_el_nmda
    W_er_nmda
    W_el_ampa
    W_er_ampa
end

Network(N_i,N_e,α,L0) = Network(N_i,N_i,N_e,
            zeros(Float32,N_i).+ V_reset,zeros(Float32,N_i).+ V_reset,zeros(Float32,N_e).+ V_reset, #potentials
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_e), #Frozem ,memories
            zeros(Float32,N_e),zeros(Float32,N_i), #activation of synapes for LMN-l -> A
            zeros(Float32,N_e),zeros(Float32,N_i), #activation of synapes for LMN-r -> A
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_i), #activation of synapes for DTN-e  DTN-e -> A
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_e),  #external inputs
            weight_init(N_i,N_e,-π+α,K_dtn_lmn),weight_init(N_i,N_i,L0), #weights
            weight_init(N_i,N_e,-π-α,K_dtn_lmn),weight_init(N_i,N_i,L0),
            weight_init(N_e,N_i,0f0,H_lmn_dtn),weight_init(N_e,N_i,0f0,H_lmn_dtn),
            weight_init(N_e,N_i,0f0,H_lmn_dtn),weight_init(N_e,N_i,0f0,H_lmn_dtn),
            )

diagonal(A::AbstractMatrix, k::Integer=0) = view(A, diagind(A, k))

function myreset!(v::AbstractArray{Float32}, frozen::AbstractArray{Float32},τref)
    pos = v .>= V_treshold
    v[pos] .= V_reset
    frozen[pos] .= -1f0*div(τref,dt)
    return pos
end

function (N::Network)(v,ext_spike_l,ext_spike_r,ext_spike_e)
    #external currents
    I_lI = gI_ampa.*(N.V_l.-V_E).*N.s_ext_ampa_l.*ext_strength_i #There is only one source of external inputs... but of much higher conductance
    I_rI = gI_ampa.*(N.V_r.-V_E).*N.s_ext_ampa_r.*ext_strength_i
    I_eE = gE_ampa.*(N.V_e.-V_E).*N.s_ext_ampa_e.*ext_strength_e
    #internal currents in the network:
    #gaba currents:
    # Inhib -> Inhib currents
    I_l_gaba = g_gaba.*(N.V_l.-V_I).*N.s_rl_gaba
    I_r_gaba = g_gaba.*(N.V_r.-V_I).*N.s_lr_gaba
    #inhib -> e curremts:
    I_e_gaba = g_gaba.*(N.V_e.-V_I).*(N.s_re_gaba .+ N.s_le_gaba)
    # e -> inhib
    I_l_nmda = g_NMDA.*(N.V_l .- V_E)./(1f0.+Mg2.*exp.(-0.062f0.*N.V_l)./3.57f0).*N.s_el_nmda
    I_r_nmda = g_NMDA.*(N.V_r .- V_E)./(1f0.+Mg2.*exp.(-0.062f0.*N.V_r)./3.57f0).*N.s_er_nmda
    I_l_ampa = g_ampa.*(N.V_l .- V_E).*N.s_el_ampa
    I_r_ampa = g_ampa.*(N.V_r .- V_E).*N.s_er_ampa

    #update the frozen variable:
    N.frozen_l .= min.(N.frozen_l .+ 1,1)
    N.frozen_r .= min.(N.frozen_r .+ 1,1)
    N.frozen_e .= min.(N.frozen_e .+ 1,1)

    #update of the potentials
    N.V_l = N.V_l .+  relu.(N.frozen_l).*(-gl.*(N.V_l .- V_leak) .- v_to_I*relu(v) .- I_lI .- I_l_gaba .- I_l_nmda).*dt/Cl
    N.V_r = N.V_r .+ relu.(N.frozen_r).*(-gr.*(N.V_r .- V_leak) .- v_to_I*anelu(v) .- I_rI .- I_r_gaba .- I_r_nmda).*dt/Cr
    N.V_e = N.V_e .+ relu.(N.frozen_e).*(-ge.*(N.V_e .-V_leak) .-I_eE .- I_e_gaba).*dt/Ce

    #see if spike occured
    spiked_l = myreset!(N.V_l,N.frozen_l,τ_ref_l)
    spiked_r = myreset!(N.V_r,N.frozen_r,τ_ref_r)
    spiked_e = myreset!(N.V_e,N.frozen_e,τ_ref_e)

    #update the synaptic variables:
    N.s_le_gaba = relu.(N.s_le_gaba + (-N.s_le_gaba*dt/τ_decay_gaba+sum(N.W_le_gaba[spiked_l,:],dims=1)[1,:]))
    N.s_re_gaba = relu.(N.s_re_gaba + (-N.s_re_gaba*dt/τ_decay_gaba+sum(N.W_re_gaba[spiked_r,:],dims=1)[1,:]))
    N.s_lr_gaba = relu.(N.s_lr_gaba + (-N.s_lr_gaba*dt/τ_decay_gaba+sum(N.W_lr_gaba[spiked_l,:],dims=1)[1,:]))
    N.s_rl_gaba = relu.(N.s_rl_gaba + (-N.s_rl_gaba*dt/τ_decay_gaba+sum(N.W_rl_gaba[spiked_r,:],dims=1)[1,:]))
    N.s_el_nmda = relu.(N.s_el_nmda + (-N.s_el_nmda*dt/τ_decay_nmda+sum(N.W_el_nmda[spiked_e,:],dims=1)[1,:]))
    N.s_er_nmda = relu.(N.s_er_nmda + (-N.s_er_nmda*dt/τ_decay_nmda+sum(N.W_er_nmda[spiked_e,:],dims=1)[1,:]))
    N.s_el_ampa = relu.(N.s_el_ampa + (-N.s_el_ampa*dt/τ_decay_ampa+sum(N.W_el_ampa[spiked_e,:],dims=1)[1,:]))
    N.s_er_ampa = relu.(N.s_er_ampa + (-N.s_er_ampa*dt/τ_decay_ampa+sum(N.W_er_ampa[spiked_e,:],dims=1)[1,:]))

    N.s_ext_ampa_l = relu.(N.s_ext_ampa_l + (-N.s_ext_ampa_l*dt/τ_decay_ampa .+ ext_spike_l))
    N.s_ext_ampa_r = relu.(N.s_ext_ampa_r + (-N.s_ext_ampa_r*dt/τ_decay_ampa .+ ext_spike_r))
    N.s_ext_ampa_e = relu.(N.s_ext_ampa_e + (-N.s_ext_ampa_e*dt/τ_decay_ampa .+ ext_spike_e))

    return spiked_e,spiked_r,spiked_l
    # return N.s_ext_ampa_i,N.V_l,N.frozen_e
end

function simulate(n::Network,T,v=0f0)
    #T: total time of the simulation
    #v: angular velocity applied to the network

    # Computes external inputs to the network:
    # simulate over time-intervals of size dt with rate ρ
    ρ_e = 18
    ρ_i = 18
    n_ext_e = 170 # 18*180 Hz total rate of spike received by the neuron.
    n_ext_i = 50 # 18*50 Hz total rate of spike received by the neuron.
    spikePerInterval_l = []
    spikePerInterval_r = []
    spikePerInterval_e = []
    raster_e_list = []
    raster_r_list = []
    raster_l_list = []

    @time for idx in ProgressBar(1:1:Int(div(T,dt)))
        probaPick_l = rand(Uniform(0,1),(n.N_l,n_ext_i))
        spikeBooleanPerInterval_l  = Float32.(probaPick_l .<= ρ_i*dt)
        spikeBooleanPerInterval_l = sum(spikeBooleanPerInterval_l,dims=2)[:,1]
        probaPick_r = rand(Uniform(0,1),(n.N_r,n_ext_i))
        spikeBooleanPerInterval_r  = Float32.(probaPick_r .<= ρ_i*dt)
        spikeBooleanPerInterval_r = sum(spikeBooleanPerInterval_r,dims=2)[:,1]
        probaPick_e = rand(Uniform(0,1),(n.N_e,n_ext_e))
        spikeBooleanPerInterval_e  = Float32.(probaPick_e .<= ρ_e*dt)
        spikeBooleanPerInterval_e = sum(spikeBooleanPerInterval_e,dims=2)[:,1]
        re,rr,rl = n(v,spikeBooleanPerInterval_l,spikeBooleanPerInterval_r,spikeBooleanPerInterval_e)
        push!(raster_e_list,re)
        push!(raster_r_list,rr)
        push!(raster_l_list,rl)
        push!(spikePerInterval_l,spikeBooleanPerInterval_l)
        push!(spikePerInterval_r,spikeBooleanPerInterval_r)
        push!(spikePerInterval_e,spikeBooleanPerInterval_e)
    end

    raster_e = Float32.(reduce(hcat,raster_e_list))
    raster_r = Float32.(reduce(hcat,raster_r_list))
    raster_l = Float32.(reduce(hcat,raster_l_list))
    spikePerInterval_e = Float32.(reduce(hcat,spikePerInterval_e))
    spikePerInterval_r = Float32.(reduce(hcat,spikePerInterval_r))
    spikePerInterval_l = Float32.(reduce(hcat,spikePerInterval_l))

    return raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r
end

function network_step_with_reorientation(N::Network,v,ext_spike_l,ext_spike_r,ext_spike_e,new_gE_ampa)
    #external currents
    I_lI = gI_ampa.*(N.V_l.-V_E).*N.s_ext_ampa_l.*ext_strength_i #There is only one source of external inputs... but of much higher conductance
    I_rI = gI_ampa.*(N.V_r.-V_E).*N.s_ext_ampa_r.*ext_strength_i
    I_eE = new_gE_ampa.*(N.V_e.-V_E).*N.s_ext_ampa_e.*ext_strength_e
    #internal currents in the network:
    #gaba currents:
    # Inhib -> Inhib currents
    I_l_gaba = g_gaba.*(N.V_l.-V_I).*N.s_rl_gaba
    I_r_gaba = g_gaba.*(N.V_r.-V_I).*N.s_lr_gaba
    #inhib -> e curremts:
    I_e_gaba = g_gaba.*(N.V_e.-V_I).*(N.s_re_gaba .+ N.s_le_gaba)
    # e -> inhib
    I_l_nmda = g_NMDA.*(N.V_l .- V_E)./(1f0.+Mg2.*exp.(-0.062f0.*N.V_l)./3.57f0).*N.s_el_nmda
    I_r_nmda = g_NMDA.*(N.V_r .- V_E)./(1f0.+Mg2.*exp.(-0.062f0.*N.V_r)./3.57f0).*N.s_er_nmda
    I_l_ampa = g_ampa.*(N.V_l .- V_E).*N.s_el_ampa
    I_r_ampa = g_ampa.*(N.V_r .- V_E).*N.s_er_ampa

    #update the frozen variable:
    N.frozen_l .= min.(N.frozen_l .+ 1,1)
    N.frozen_r .= min.(N.frozen_r .+ 1,1)
    N.frozen_e .= min.(N.frozen_e .+ 1,1)

    #update of the potentials
    N.V_l = N.V_l .+  relu.(N.frozen_l).*(-gl.*(N.V_l .- V_leak) .- v_to_I*relu(v) .- I_lI .- I_l_gaba .- I_l_nmda).*dt/Cl
    N.V_r = N.V_r .+ relu.(N.frozen_r).*(-gr.*(N.V_r .- V_leak) .- v_to_I*anelu(v) .- I_rI .- I_r_gaba .- I_r_nmda).*dt/Cr
    N.V_e = N.V_e .+ relu.(N.frozen_e).*(-ge.*(N.V_e .-V_leak) .-I_eE .- I_e_gaba).*dt/Ce

    #see if spike occured
    spiked_l = myreset!(N.V_l,N.frozen_l,τ_ref_l)
    spiked_r = myreset!(N.V_r,N.frozen_r,τ_ref_r)
    spiked_e = myreset!(N.V_e,N.frozen_e,τ_ref_e)

    #update the synaptic variables:
    N.s_le_gaba = relu.(N.s_le_gaba + (-N.s_le_gaba*dt/τ_decay_gaba+sum(N.W_le_gaba[spiked_l,:],dims=1)[1,:]))
    N.s_re_gaba = relu.(N.s_re_gaba + (-N.s_re_gaba*dt/τ_decay_gaba+sum(N.W_re_gaba[spiked_r,:],dims=1)[1,:]))
    N.s_lr_gaba = relu.(N.s_lr_gaba + (-N.s_lr_gaba*dt/τ_decay_gaba+sum(N.W_lr_gaba[spiked_l,:],dims=1)[1,:]))
    N.s_rl_gaba = relu.(N.s_rl_gaba + (-N.s_rl_gaba*dt/τ_decay_gaba+sum(N.W_rl_gaba[spiked_r,:],dims=1)[1,:]))
    N.s_el_nmda = relu.(N.s_el_nmda + (-N.s_el_nmda*dt/τ_decay_nmda+sum(N.W_el_nmda[spiked_e,:],dims=1)[1,:]))
    N.s_er_nmda = relu.(N.s_er_nmda + (-N.s_er_nmda*dt/τ_decay_nmda+sum(N.W_er_nmda[spiked_e,:],dims=1)[1,:]))
    N.s_el_ampa = relu.(N.s_el_ampa + (-N.s_el_ampa*dt/τ_decay_ampa+sum(N.W_el_ampa[spiked_e,:],dims=1)[1,:]))
    N.s_er_ampa = relu.(N.s_er_ampa + (-N.s_er_ampa*dt/τ_decay_ampa+sum(N.W_er_ampa[spiked_e,:],dims=1)[1,:]))

    N.s_ext_ampa_l = relu.(N.s_ext_ampa_l + (-N.s_ext_ampa_l*dt/τ_decay_ampa .+ ext_spike_l))
    N.s_ext_ampa_r = relu.(N.s_ext_ampa_r + (-N.s_ext_ampa_r*dt/τ_decay_ampa .+ ext_spike_r))
    N.s_ext_ampa_e = relu.(N.s_ext_ampa_e + (-N.s_ext_ampa_e*dt/τ_decay_ampa .+ ext_spike_e))

    return spiked_e,spiked_r,spiked_l
    # return N.s_ext_ampa_i,N.V_l,N.frozen_e
end

function simulate_external_inputs(n::Network,T,θR,v=0f0)
    #T: total time of the simulation
    #v: angular velocity applied to the network

    # Computes external inputs to the network:
    # simulate over time-intervals of size dt with rate ρ
    ρ_e = 18
    ρ_i = 18
    n_ext_e = 170 # 18*180 Hz total rate of spike received by the neuron.
    n_ext_i = 50 # 18*50 Hz total rate of spike received by the neuron.
    spikePerInterval_l = []
    spikePerInterval_r = []
    spikePerInterval_e = []
    raster_e_list = []
    raster_r_list = []
    raster_l_list = []
    R = 0f0

    #parameters for this experiment:
    σ_R = 45f0*π/180f0
    R_0 = 1.3f0
    τ_decay_R = 0.1f0

    θ_bin_e = Float32.(range(-π,stop=π,length=n.N_e))
    @time for idx in ProgressBar(1:1:Int(div(T,dt)))
        probaPick_l = rand(Uniform(0,1),(n.N_l,n_ext_i))
        spikeBooleanPerInterval_l  = Float32.(probaPick_l .<= ρ_i*dt)
        spikeBooleanPerInterval_l = sum(spikeBooleanPerInterval_l,dims=2)[:,1]
        probaPick_r = rand(Uniform(0,1),(n.N_r,n_ext_i))
        spikeBooleanPerInterval_r  = Float32.(probaPick_r .<= ρ_i*dt)
        spikeBooleanPerInterval_r = sum(spikeBooleanPerInterval_r,dims=2)[:,1]
        probaPick_e = rand(Uniform(0,1),(n.N_e,n_ext_e))
        spikeBooleanPerInterval_e  = Float32.(probaPick_e .<= ρ_e*dt)
        spikeBooleanPerInterval_e = sum(spikeBooleanPerInterval_e,dims=2)[:,1]

        if idx*dt>=1f0
            R = R_0*exp(-(Float32(idx*dt)-1f0)/τ_decay_R) #expential decay after the onset of the perturbation.
        end
        print(R)
        new_ge_ampa = gE_ampa.*(1f0.+R.*exp.(-angleDiff.(θR.-θ_bin_e).^2 ./(2f0*σ_R^2)))
        re,rr,rl = network_step_with_reorientation(n,v,spikeBooleanPerInterval_l,spikeBooleanPerInterval_r,spikeBooleanPerInterval_e,new_ge_ampa)
        push!(raster_e_list,re)
        push!(raster_r_list,rr)
        push!(raster_l_list,rl)
        push!(spikePerInterval_l,spikeBooleanPerInterval_l)
        push!(spikePerInterval_r,spikeBooleanPerInterval_r)
        push!(spikePerInterval_e,spikeBooleanPerInterval_e)
    end

    raster_e = Float32.(reduce(hcat,raster_e_list))
    raster_r = Float32.(reduce(hcat,raster_r_list))
    raster_l = Float32.(reduce(hcat,raster_l_list))
    spikePerInterval_e = Float32.(reduce(hcat,spikePerInterval_e))
    spikePerInterval_r = Float32.(reduce(hcat,spikePerInterval_r))
    spikePerInterval_l = Float32.(reduce(hcat,spikePerInterval_l))

    return raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r
end


#For explorations (beyond the paper)

function noisyRecursiveWeights(N1::Int,N2::Int,L0::Float32,σ_0)
    w = Float32.(rand(Normal(0f0,σ_0),(N1,N2))) .+ L0
end
Network_noisy_recursive_Weights(N_i,N_e,α,L0,σ_0) = Network(N_i,N_i,N_e,
            zeros(Float32,N_i).+ V_reset,zeros(Float32,N_i).+ V_reset,zeros(Float32,N_e).+ V_reset, #potentials
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_e), #Frozem ,memories
            zeros(Float32,N_e),zeros(Float32,N_i), #activation of synapes for LMN-l -> A
            zeros(Float32,N_e),zeros(Float32,N_i), #activation of synapes for LMN-r -> A
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_i), #activation of synapes for DTN-e  DTN-e -> A
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_e),  #external inputs
            weight_init(N_i,N_e,-π+α,K_dtn_lmn),noisyRecursiveWeights(N_i,N_i,L0,σ_0), #weights
            weight_init(N_i,N_e,-π-α,K_dtn_lmn),noisyRecursiveWeights(N_i,N_i,L0,σ_0),
            weight_init(N_e,N_i,0f0,H_lmn_dtn),weight_init(N_e,N_i,0f0,H_lmn_dtn),
            weight_init(N_e,N_i,0f0,H_lmn_dtn),weight_init(N_e,N_i,0f0,H_lmn_dtn),
            )
Network_spatialy_tuned_inhibinhib(N_i,N_e,α,L0,α_2,σ_0) = Network(N_i,N_i,N_e,
            zeros(Float32,N_i).+ V_reset,zeros(Float32,N_i).+ V_reset,zeros(Float32,N_e).+ V_reset, #potentials
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_e), #Frozem ,memories
            zeros(Float32,N_e),zeros(Float32,N_i), #activation of synapes for LMN-l -> A
            zeros(Float32,N_e),zeros(Float32,N_i), #activation of synapes for LMN-r -> A
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_i), #activation of synapes for DTN-e  DTN-e -> A
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_e),  #external inputs
            weight_init(N_i,N_e,-π+α,K_dtn_lmn).+Float32.(rand(Normal(0f0,σ_0),(N_i,N_i))),weight_init(N_i,N_i,α_2,H_lmn_dtn).+Float32.(rand(Normal(0f0,σ_0),(N_i,N_i))), #weights
            weight_init(N_i,N_e,-π-α,K_dtn_lmn).+Float32.(rand(Normal(0f0,σ_0),(N_i,N_i))),weight_init(N_i,N_i,α_2,H_lmn_dtn).+Float32.(rand(Normal(0f0,σ_0),(N_i,N_i))),
            weight_init(N_e,N_i,0f0,H_lmn_dtn),weight_init(N_e,N_i,0f0,H_lmn_dtn),
            weight_init(N_e,N_i,0f0,H_lmn_dtn),weight_init(N_e,N_i,0f0,H_lmn_dtn),
            )



Network_spatialy_tuned_inhibinhib(N_i,N_e,α,L0,α_2,σ_0) = Network(N_i,N_i,N_e,
            zeros(Float32,N_i).+ V_reset,zeros(Float32,N_i).+ V_reset,zeros(Float32,N_e).+ V_reset, #potentials
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_e), #Frozem ,memories
            zeros(Float32,N_e),zeros(Float32,N_i), #activation of synapes for LMN-l -> A
            zeros(Float32,N_e),zeros(Float32,N_i), #activation of synapes for LMN-r -> A
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_i), #activation of synapes for DTN-e  DTN-e -> A
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_e),  #external inputs
            weight_init(N_i,N_e,-π+α,K_dtn_lmn).+Float32.(rand(Normal(0f0,σ_0),(N_i,N_i))),weight_init(N_i,N_i,α_2,H_lmn_dtn).+Float32.(rand(Normal(0f0,σ_0),(N_i,N_i))), #weights
            weight_init(N_i,N_e,-π-α,K_dtn_lmn).+Float32.(rand(Normal(0f0,σ_0),(N_i,N_i))),weight_init(N_i,N_i,α_2,H_lmn_dtn).+Float32.(rand(Normal(0f0,σ_0),(N_i,N_i))),
            weight_init(N_e,N_i,0f0,H_lmn_dtn),weight_init(N_e,N_i,0f0,H_lmn_dtn),
            weight_init(N_e,N_i,0f0,H_lmn_dtn),weight_init(N_e,N_i,0f0,H_lmn_dtn),
            )
K1 = 1.1f0
σ_l = 30f0*π/180f0
K2_dtn_lmn(θ) = 2*K1*exp(-8*cos(2*θ+π)/(σ_l)^2/2)/exp(8/(2*(σ_l)^2))
H1 = 0.065f0
σ_e = 80f0*π/180f0
H2_lmn_dtn(θ) = 2*H1*exp(-10*cos(2*θ+π)/(σ_e)^2/2)/exp(10/(2*(σ_e)^2))
Network_new_weights(N_i,N_e,α,L0) = Network(N_i,N_i,N_e,
            zeros(Float32,N_i).+ V_reset,zeros(Float32,N_i).+ V_reset,zeros(Float32,N_e).+ V_reset, #potentials
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_e), #Frozem ,memories
            zeros(Float32,N_e),zeros(Float32,N_i), #activation of synapes for LMN-l -> A
            zeros(Float32,N_e),zeros(Float32,N_i), #activation of synapes for LMN-r -> A
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_i), #activation of synapes for DTN-e  DTN-e -> A
            zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_e),  #external inputs
            weight_init(N_i,N_e,-π+α,K2_dtn_lmn),weight_init(N_i,N_i,L0), #weights
            weight_init(N_i,N_e,-π-α,K2_dtn_lmn),weight_init(N_i,N_i,L0),
            weight_init(N_e,N_i,0f0,H2_lmn_dtn),weight_init(N_e,N_i,0f0,H2_lmn_dtn),
            weight_init(N_e,N_i,0f0,H2_lmn_dtn),weight_init(N_e,N_i,0f0,H2_lmn_dtn),
            )
function simulate(n::Network,T,v::AbstractArray{Float32})
    #T: total time of the simulation
    #v: angular velocity applied to the network, array which steps should be of dt!
    print("simulating with speed array")
    # Computes external inputs to the network:
    # simulate over time-intervals of size dt with rate ρ
    ρ_e = 18
    ρ_i = 18
    n_ext_e = 170 # 18*170 Hz total rate of spike received by the neuron.
    n_ext_i = 50 # 18*50 Hz total rate of spike received by the neuron.
    raster_e_list = []
    @time for idx in ProgressBar(1:1:Int(div(T,dt)))
        probaPick_l = rand(Uniform(0,1),(n.N_l,n_ext_i))
        spikeBooleanPerInterval_l  = Float32.(probaPick_l .<= ρ_i*dt)
        spikeBooleanPerInterval_l = sum(spikeBooleanPerInterval_l,dims=2)[:,1]
        probaPick_r = rand(Uniform(0,1),(n.N_r,n_ext_i))
        spikeBooleanPerInterval_r  = Float32.(probaPick_r .<= ρ_i*dt)
        spikeBooleanPerInterval_r = sum(spikeBooleanPerInterval_r,dims=2)[:,1]
        probaPick_e = rand(Uniform(0,1),(n.N_e,n_ext_e))
        spikeBooleanPerInterval_e  = Float32.(probaPick_e .<= ρ_e*dt)
        spikeBooleanPerInterval_e = sum(spikeBooleanPerInterval_e,dims=2)[:,1]
        re,rr,rl = n(v[idx],spikeBooleanPerInterval_l,spikeBooleanPerInterval_r,spikeBooleanPerInterval_e)
        push!(raster_e_list,re)
    end

    raster_e = Float32.(reduce(hcat,raster_e_list))
    return raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r
end
