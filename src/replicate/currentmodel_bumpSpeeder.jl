# We wish to obtain Interneurons which tuning curve is the fourier component of the tuning curve of an HD cells
# A first solution is a hard-wired model,
#   Indeed, if we make the bump goes n time faster in the inhibitory ring, then the overall tuning curve will be the same
#   but integrated three time faster...
# Now a crucial point is that our inhibitory interneurons seems not to be tuned to the HD signal, and this modulation seems to appear only slightly above them.
# If we assume that the bump moves throught the external inputs, and the connection to pyr in the local ring homogeneous
#   we could maybe obtain the same effect, but attenuated.


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
gl = 20f0
τl = 0.01f0 #10 ms 🔥
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
V_leak = -70f0 #-70mV
V_treshold = -50f0 # -50mV
V_reset = -55f0 # -55mV

v_to_I = 5.5f0 #conversion from speed input into current: not provided in the paper. We obtained it manually with a few trials, the goal is just to reproduce the shape of the figures for figure 9
ext_strength_i =  1f0#3.5f0/1.62f0 # strength factor for the synaptic connection between background inputs and neurons in the network
ext_strength_e = 1f0#5.07f0/2.08f0
#simulation parameters
dt = 0.0005f0 # 0.5 ms

#weight parameters
angleDiff(e) = min(abs(e),2f0*π-abs(e))

H1 = 0.065f0
σ_e = 40f0*π/180f0
H_lmn_dtn(θ) = H1*exp(-(-angleDiff(ω*θ)/σ_e)^2/2)

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
    N_l #Number of FS inhibitory in Post-sub
    N_e #Number of excitatory in AD
    V_l #membrane potentials
    V_e
    #frozen memories: counters that starts at -div(τ_ref,dt) and are increased by one until they reached the value 1 where they are not increased anymore.
    # The Neuron dynamic then starts again.
    frozen_l
    frozen_e
    #activation of synapes for LMN-l -> A
    s_le_gaba # 🔥 these variables are vector where each elements is the sum of activations from synapses of pop l onto pop e 🔥
    #activation of synapes for DTN-e -> A
    s_el_nmda
    s_el_ampa
    #external inputs: AMPA synaptic receptors (For Ii and IE)
    s_ext_ampa_l
    s_ext_ampa_e

    #Weights
    #Weights of synapes for DTN-e -> A
    W_el_nmda
    W_el_ampa
end

Network(N_i,N_e,α,L0) = Network(N_i,N_e,
            zeros(Float32,N_i).+ V_reset,zeros(Float32,N_e).+ V_reset, #potentials
            zeros(Float32,N_i),zeros(Float32,N_e), #Frozem ,memories
            zeros(Float32,N_e), #activation of synapes for LMN-l -> A
            zeros(Float32,N_i),zeros(Float32,N_i), #activation of synapes for DTN-e  DTN-e -> A
            zeros(Float32,N_i),zeros(Float32,N_e),  #external inputs
            weight_init(N_e,N_i,0f0,H_lmn_dtn),
            weight_init(N_e,N_i,0f0,H_lmn_dtn)
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
    I_eE = gE_ampa.*(N.V_e.-V_E).*N.s_ext_ampa_e.*ext_strength_e

    #internal currents in the network:
    # e -> inhib
    I_l_nmda = g_NMDA.*(N.V_l .- V_E)./(1f0.+Mg2.*exp.(-0.062f0.*N.V_l)./3.57f0).*N.s_el_nmda
    I_l_ampa = g_ampa.*(N.V_l .- V_E).*N.s_el_ampa

    #update the frozen variable:
    N.frozen_l .= min.(N.frozen_l .+ 1,1)
    N.frozen_e .= min.(N.frozen_e .+ 1,1)

    #update of the potentials
    N.V_l = N.V_l .+  relu.(N.frozen_l).*(-gl.*(N.V_l .- V_leak) .- v_to_I*relu(v) .- I_lI .- I_l_gaba .- I_l_nmda).*dt/Cl
    N.V_e = N.V_e .+ relu.(N.frozen_e).*(-ge.*(N.V_e .-V_leak) .-I_eE .-).*dt/Ce

    #see if spike occured
    spiked_l = myreset!(N.V_l,N.frozen_l,τ_ref_l)
    spiked_e = myreset!(N.V_e,N.frozen_e,τ_ref_e)

    #update the synaptic variables:
    N.s_el_nmda = relu.(N.s_el_nmda + (-N.s_el_nmda*dt/τ_decay_nmda+sum(N.W_el_nmda[spiked_e,:],dims=1)[1,:]))
    N.s_el_ampa = relu.(N.s_el_ampa + (-N.s_el_ampa*dt/τ_decay_ampa+sum(N.W_el_ampa[spiked_e,:],dims=1)[1,:]))

    N.s_ext_ampa_l = relu.(N.s_ext_ampa_l + (-N.s_ext_ampa_l*dt/τ_decay_ampa .+ ext_spike_l))
    N.s_ext_ampa_e = relu.(N.s_ext_ampa_e + (-N.s_ext_ampa_e*dt/τ_decay_ampa .+ ext_spike_e))

    return spiked_e,spiked_l
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
