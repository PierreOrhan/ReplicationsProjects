#Replicate the 2014 Widloski/Fiete paper
using Distributions
using Flux

CV = 0.5f0
m = Int(1/(CV^2))
τs = 0.0010f0 #time constant of neural-response: 10 ms
dt = 0.0005f0 #time-step for numerical integration: 0.5ms
wₒ = 0.001f0 #initial weights
gₒ = 50f0 # constant bias terms commons in all cells
gₒ2 = 15f0 # constant bias terms commons in all cells
η = 0.015f0 # learning rates
γ_ii = 7f0
γ_ei = 2f0
γ_ie = 1f0
τ_stdp = 0.0012f0 #time cst for stdp: 12 ms
α_stdp = 1f0 #cst for stdp: 1
A = 1.2f0 #cst for stdp kernels
B = 0.5f0 #cst for stdp kernels

weightInit(in,out,wₒ) = Float32.(rand(Uniform(min(0,wₒ),max(0,wₒ)),(out,in)))
nelu(x) = -relu(-1f0 .* x)

mutable struct OneDstdpLayer
    N_i #Number of inhibitory
    N_e #Number of excitatory
    s_i #activation of synapes
    s_e
    I_i #synaptic current
    I_e
    r_i #instantaneous firing rate
    r_e
    #weights
    W_ie #inhibitory to excitatory
    W_ei #excitatory to inhibitory
    W_ii #inhibitory to inhibitory connections
    # parameters
    isStochasticDynamics #boolean
    #  Integrals of learning kernels
    #   These integrals can be computed iteratively because of the exponentials kernels!
    #       the effect of considering initially or not the effect of the kernels
    #       adds a constant term times an exponentially decreasing term...
    #           ==> we can initialize the integral at 0...
    Icausal_e
    Iacausal_ei
    Icausal_i
    Iacausal_ie
    Iacausal_ii
    #traces terms
    kcausal_e # = dt/(2*α_stdp*τ_stdp)
    kacausal_e # = dt/(1.5*α_stdp*τ_stdp)
    kcausal_i # = dt/(2*α_stdp*τ_stdp))
    kacausal_i # = dt/(α_stdp*τ_stdp)

    #memories for the spike trains decimation
    spike_mem_e
    spike_mem_i
end

OneDstpLayer(N_i,N_e,dynamic) = OneDstdpLayer( N_i,N_e,
            zeros(Float32,N_i),zeros(Float32,N_e),zeros(Float32,N_i),zeros(Float32,N_e),zeros(Float32,N_i),zeros(Float32,N_e),
            weightInit(N_i,N_e,-wₒ),weightInit(N_e,N_i,wₒ),weightInit(N_i,N_i,-wₒ),dynamic,
            zeros(Float32,N_e),zeros(Float32,N_i),zeros(Float32,N_i),zeros(Float32,N_e),zeros(Float32,N_i),
            dt/(2f0*α_stdp*τ_stdp),dt/(1.5f0*α_stdp*τ_stdp),dt/(2f0*α_stdp*τ_stdp),dt/(α_stdp*τ_stdp),
            0f0,0f0)

diagonal(A::AbstractMatrix, k::Integer=0) = view(A, diagind(A, k))

function (l::OneDstdpLayer)(inputs_e,inputs_i,xshunt_e,xshunt_i,env_e,env_i)

    #update the total synaptic current: computes synaptic current at time t
    l.I_e = env_e.*xshunt_e.*(l.W_ie*l.s_i + inputs_e .+ gₒ) .+ env_e.*gₒ2
    l.I_i = env_i.*xshunt_i.*(l.W_ei*l.s_e + l.W_ii*l.s_i + inputs_i .+ gₒ) .+ env_i.*gₒ2
    #update the instantaneous firing rate (s⁻¹): computes firing rate at time t. Approximated to be constant in [t,t+dt]
    l.r_e = relu.(l.I_e)
    l.r_i = relu.(l.I_i)

    #update the synapse activation s
    # uses σ: If the dynamic is not stochastic σ is a rate (s⁻¹),
    # otherwise it is the "total spiking": number of spike in [t,td+dt)

    #samples train over m*dt, here CV = 1/√m, m=4,
    if l.isStochasticDynamics
        train_e = subPoisson(l.r_e,m,dt,size(l.r_e,1))
        full_train_e = l.spike_mem_e .+ sum(train_e,dims=1)[1,:]
        nb_spike_e = Float32.(div.(full_train_e,m)) #nb of spikes in [t,t+dt]
        l.spike_mem_e = (full_train_e .% m)

        train_i = subPoisson(l.r_i,m,dt,size(l.r_i,1))
        full_train_i = l.spike_mem_i .+ sum(train_i,dims=1)[1,:]
        nb_spike_i = Float32.(div.(full_train_i,m))
        l.spike_mem_i = (full_train_i .% m)
    else
        nb_spike_e = 0f0
        nb_spike_i = 0f0
    end

    σ_e = l.isStochasticDynamics ? nb_spike_e : l.r_e*dt
    σ_i = l.isStochasticDynamics ? nb_spike_i : l.r_i*dt
    #update the activation of synapses (no unit): 🔥 Problem here?? should it be σ_i*dt/τs ?
    l.s_i = σ_i + l.s_i*(1f0-dt/τs)
    l.s_e = σ_e + l.s_e*(1f0-dt/τs)

    #update of the STDP traces, using the presynaptic spike trains:
    l.Icausal_e = l.Icausal_e*(1f0-l.kcausal_e) + A*σ_e
    l.Icausal_i = l.Icausal_i*(1f0-l.kcausal_i) + B*σ_i
    l.Iacausal_ei = l.Iacausal_ei*(1f0-l.kacausal_e) - σ_i
    l.Iacausal_ie = l.Iacausal_ie*(1f0-l.kacausal_i) - σ_e
    l.Iacausal_ii = l.Iacausal_ii*(1f0-l.kacausal_i) - σ_i

    #Update of the weights: 🔥 W_ie corresponds to W_P'P=W_EI in the paper 🔥
    # ❗ e.g: W_ie is the weight matrix for inhibitory inhibiting excitatory cells
    # ❗ so W_ie is of size (l.N_e,l.N_i)

    # Weights update are made according to the following rule which is equivalent to STDP:
    # dw/dt|_t = ∑_(tpre<t)∑_(tpost<t) K(tpre-tpost) , here K is given by the integral in the paper
    # Now if we discretize this equation we obtain the following update rules:
    # When a post_synaptic spike
    # w(t+dt) = w(t) + ∑_tpre<t A_pre exp(-(t-t_pre)/τ_pre)  We can obtain the right element with an exponential filter, calling it a_pre, we obtain:

    #   When a post-synaptic spike occurs: w → w + apost
    #   When a pres-synaptic spike occurs: w → w + apre
    # apost and apre are STDP traces of a pre and post-synaptic activity. They are defined just above under the name Icausal_? ↔ apre_? and Iacausal_? ↔ apost_?.
    #   When working with a rate model, we always apply the update but scaled by the rate*dt
    # ❗ e.g: for W_ie, spiking model:
    # Let us focus on the synapse l→g where l is the index of inhibitory neurons and g of excitatory neurons
    #  When only a post synaptic spike occurs in neurons g, σ_e[g] =1 → (W_ie)[g,l] → (W_ie)[g,l] + 1*apre_l = (W_ie)[g,l] + 1*(Icausal_i)[l]
    #  When only a pre synaptic spike occurs in neurons g, σ_i[l] =1 → (W_ie)[g,l] → (W_ie)[g,l] + 1*apost_g = (W_ie)[g,l] + Iacausal_ie[g]
    # It is important to have 2 traces for each synapses! Nonetheless, we can summarize them by 2 traces for each weight type in each neurons.
    # This is because in those models we assyme the neuron spiking spreads to all of its synapses.

    l.W_ie = nelu.(l.W_ie + η*γ_ei*(σ_e*transpose(l.Icausal_i) + l.Iacausal_ie*transpose(σ_i)))
    l.W_ei = relu.(l.W_ei + η*γ_ie*(σ_i*transpose(l.Icausal_e) + l.Iacausal_ei*transpose(σ_e)))
    l.W_ii = nelu.(l.W_ii + η*γ_ii*(σ_i*transpose(l.Icausal_i) + l.Iacausal_ii*transpose(σ_i)))
    diagonal(l.W_ii) .= 0f0 # No self-inhibition! (also called autaptic connections)
end

Flux.@treelike OneDstdpLayer

function subPoisson(ρ,m,dt,NbNeurons)
    # we sample from a inhomogeneous process with rate ρ(t) and CV = ρ/√(m)
    # here the sampling is made over dt
    # CV is the coefficient of variation of the interspike interval

    # Note 1: there is two strategy to generate a spike train
    #  Either consider fix time bin and assign a spike to this time bin with proba ρ*m*dt
    #  Or choose interspike intervals randomly from the exponential distribution...
    # but since we will use the spike trains with discrete time bin, it is more wise to use the first approach
    # as the second falls back to it in this case.

    # Note 2: the coefficient of variation highlights a switch from a Poisson process to a Renewal intervals
    #  The interspike interval becomes gamma of order m, where CV = σ/ρ = 1/√m
    # To do so we need to remove from a poisson-generated spike train every mth spike
    # Another solution is to simulate m trains with rate divided by m, merge them and keep solely the m-th spikes.

    # To obtain a real subPoisson process,
   # one need to work with a full train that we decimate.
   # As such, the result of this function need to be added at the end of a full
   # train, which we decimate progressively

   # A solution is to keep a count of the number of spikes in the previous bin
   #First strategy:
   probaPick = rand(Uniform(0,1),(Int(m),NbNeurons))
   # simulate over time-intervals of size dt/m with rate ρ*m
   spikeBooleanPerInterval = zeros(Int,(Int(m),NbNeurons))
   for e in 1:1:m
       spikeBooleanPerInterval[e,:] = Int.(probaPick[e,:] .<= ρ*dt)
   end
   return spikeBooleanPerInterval
end
#gpu version
function subPoisson(ρ::CuArray,m,dt,NbNeurons)
   probaPick = gpu(rand(Uniform(0,1),(Int(m),NbNeurons)))
   # simulate over time-intervals of size dt/m with rate ρ*m
   spikeBooleanPerInterval = gpu(zeros((Int(m),NbNeurons)))
   for e in 1:1:m
       spikeBooleanPerInterval[e,:] = probaPick[e,:] .<= ρ*dt
   end
   return spikeBooleanPerInterval
end

function sampling(l::OneDstdpLayer,inputs_e,inputs_i,xshunt_e,xshunt_i)
    l.I_e = xshunt_e.*(l.W_ie*l.s_i + inputs_e .+ gₒ) .+ gₒ2
    l.I_i = xshunt_i.*(l.W_ei*l.s_e + l.W_ii*l.s_i + inputs_i .+ gₒ) .+ gₒ2
    #update the instantaneous firing rate (s⁻¹): computes firing rate at time t. Approximated to be constant in [t,t+dt]
    l.r_e = relu.(l.I_e)
    l.r_i = relu.(l.I_i)

    #update the synapse activation s
    # uses σ: If the dynamic is stochastic σ is a rate (s⁻¹), otherwise it is the "total spiking": number of spike in [t,td+dt)

    #samples train over m*dt, here CV = 1/√m, m=4,
    if l.isStochasticDynamics
        train_e = subPoisson(l.r_e,m,dt,size(l.r_e,1))
        full_train_e = l.spike_mem_e .+ sum(train_e,dims=1)[1,:]
        nb_spike_e = Float32.(div.(full_train_e,m)) #nb of spikes in [t,t+dt]
        l.spike_mem_e = (full_train_e .% m)

        train_i = subPoisson(l.r_i,m,dt,size(l.r_i,1))
        full_train_i = l.spike_mem_i .+ sum(train_i,dims=1)[1,:]
        nb_spike_i = Float32.(div.(full_train_i,m))
        l.spike_mem_i = (full_train_i .% m)
    else
        nb_spike_e = 0f0
        nb_spike_i = 0f0
    end
    σ_e = l.isStochasticDynamics ? nb_spike_e : l.r_e*dt
    σ_i = l.isStochasticDynamics ? nb_spike_i : l.r_i*dt
    #update the activation of synapses (no unit):
    l.s_i = σ_i + l.s_i*(1-dt/τs)
    l.s_e = σ_e + l.s_e*(1-dt/τs)

    return σ_e,σ_i
end
function reset_synapse(l::OneDstdpLayer)
    l.s_i[:] = l.s_i .* 0f0
    l.s_e[:] = l.s_e .* 0f0
end
function reset_synapse(l::OneDstdpLayer,si,se)
    l.s_i[:] = si
    l.s_e[:] = se
end
