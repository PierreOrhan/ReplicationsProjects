using Flux
using PyPlot
using Interpolations
using LinearAlgebra
using BenchmarkTools
using ProgressBars
using Distributions
using Roots
using ForwardDiff
using DifferentialEquations

path = joinpath("src","replicate","replicateBrunel2004")

dt = 0.0005f0 # time-step : 5ms
τ = 0.0010f0 # synaptic time constant (from Fiete 2014...)
T = 10 #total duration : 10sc

#Replication of figure 2
Iᵢ = 1
#model definition:
function solve(Ii,L0,f0, ϕ = relu)
    f = zeros(Float32,(Int(div(T,dt)),2))
    f[1,:] =f0
    for idx in 1:1:size(f,1)-1
        f[idx+1,1] = ϕ(Ii-L0*f[idx,2])*dt/τ + f[idx,1]*(1-dt/τ)
        f[idx+1,2] = ϕ(Ii-L0*f[idx,1])*dt/τ + f[idx,2]*(1-dt/τ)
    end
    return f[end,:]
end
L0 = 0f0:0.01f0:2f0
#simulate the relu case, with distinct initialization:
rates_relu_distinc_init = reduce(hcat,map(l0->solve(1,l0,rand(Uniform(0,1),2)),L0))
function getsim(nb=2)
    f = rand(Uniform(0f0,1f0),1)[1]
    return [f for _ in 1:1:nb]
end
#simulate the relu case, with similar initialization (the network stays on the saddle point)
rates_relu_same_init = reduce(hcat,map(l0->solve(1,l0,getsim()),L0))

#quad case
quad(I) = I<0f0 ? 0f0 : I<=1f0 ? I*I : 2f0*√(I-3f0/4f0)
rates_quad_distinct_init = reduce(hcat,map(l0->solve(1,l0,rand(Uniform(0,1),2),quad),L0))
rates_quad_same_init = reduce(hcat,map(l0->solve(1,l0,getsim(),quad),L0))

#varying external input with quad case
Ii = 0f0:0.01f0:5f0
rates_quad_distinct_init_varying_inputs = reduce(hcat,map(I->solve(I,1,rand(Uniform(0,1),2),quad),Ii))
rates_quad_same_init_varying_inputs = reduce(hcat,map(I->solve(I,1,getsim(),quad),Ii))

#figure plot
fig,ax = plt.subplots(1,3,figsize=(10,10))
ax[1].scatter(L0[100:end],rates_relu_distinc_init[1,100:end],color=(0,0,0),s=0.2)
ax[1].scatter(L0[100:end],rates_relu_distinc_init[2,100:end],color=(0,0,0),s=0.2)
ax[1].scatter(L0[1:99],rates_relu_same_init[1,1:99],color=(0,0,0),s=0.2)
ax[1].scatter(L0[1:99],rates_relu_same_init[2,1:99],color=(0,0,0),s=0.2)
ax[1].scatter(L0[99:5:end],rates_relu_same_init[1,99:5:end],color=(0,0,0),s=0.2)
ax[1].scatter(L0[99:5:end],rates_relu_same_init[2,99:5:end],color=(0,0,0),s=0.2)
ax[1].set_xlabel("inhibitory coupling")
ax[1].set_ylabel("rate")
ax[1].set_aspect("equal")
ax[2].scatter(L0[1:end],rates_quad_distinct_init[1,1:end],color=(0,0,0),s=0.2)
ax[2].scatter(L0[1:end],rates_quad_distinct_init[2,1:end],color=(0,0,0),s=0.2)
ax[2].scatter(L0[1:99],rates_quad_same_init[1,1:99],color=(0,0,0),s=0.2)
ax[2].scatter(L0[1:99],rates_quad_same_init[2,1:99],color=(0,0,0),s=0.2)
ax[2].scatter(L0[99:5:end],rates_quad_same_init[1,99:5:end],color=(0,0,0),s=0.2)
ax[2].scatter(L0[99:5:end],rates_quad_same_init[2,99:5:end],color=(0,0,0),s=0.2)
ax[2].set_xlabel("inhibitory coupling")
ax[2].set_aspect("equal")
ax[3].scatter(Ii[1:end],rates_quad_distinct_init_varying_inputs[1,1:end],color=(0,0,0),s=0.2)
ax[3].scatter(Ii[1:end],rates_quad_distinct_init_varying_inputs[2,1:end],color=(0,0,0),s=0.2)
ax[3].scatter(Ii[1:99],rates_quad_same_init_varying_inputs[1,1:99],color=(0,0,0),s=0.2)
ax[3].scatter(Ii[1:99],rates_quad_same_init_varying_inputs[2,1:99],color=(0,0,0),s=0.2)
ax[3].scatter(Ii[1:5:end],rates_quad_same_init_varying_inputs[1,1:5:end],color=(0,0,0),s=0.2)
ax[3].scatter(Ii[1:5:end],rates_quad_same_init_varying_inputs[2,1:5:end],color=(0,0,0),s=0.2)
ax[3].set_xlabel("External input")
ax[3].set_ylabel("rate")
ax[3].set_aspect("equal")
display(fig)
fig.savefig(joinpath(path,"figures","2B.png"))


##Second model: two excitatory and 2 inhibitory units

#figure 3:
# Note that the same synaptic time cst is used .. (questionable?)
function solve2(Ie,K1,H1,f0::Array{Float32,1}, ϕ = relu)
    f = zeros(Float32,(Int(div(T,dt)),4))
    f[1,:] =f0
    for idx in 1:1:size(f,1)-1
        #excitatory
        f[idx+1,1] = ϕ(Ie-K1*f[idx,4])*dt/τ + f[idx,1]*(1-dt/τ)
        f[idx+1,2] = ϕ(Ie-K1*f[idx,3])*dt/τ + f[idx,2]*(1-dt/τ)
        #inhibitory
        f[idx+1,3] = ϕ(H1*f[idx,1])*dt/τ + f[idx,3]*(1-dt/τ)
        f[idx+1,4] = ϕ(H1*f[idx,2])*dt/τ + f[idx,4]*(1-dt/τ)
    end
    return f[end,:]
end

X = 0f0:0.01f0:5f0
@time resH = reduce(hcat,map(x->solve2(1f0,1f0,x,Float32.(rand(Uniform(0f0,1f0),4)),quad),X))
@time resK = reduce(hcat,map(x->solve2(1f0,x,1f0,Float32.(rand(Uniform(0f0,1f0),4)),quad),X))
@time resI = reduce(hcat,map(x->solve2(x,1f0,1f0,Float32.(rand(Uniform(0f0,1f0),4)),quad),X))
@time resH2 = reduce(hcat,map(x->solve2(1f0,1f0,x,Float32.(getsim(4)),quad),X))
@time resK2 = reduce(hcat,map(x->solve2(1f0,x,1f0,Float32.(getsim(4)),quad),X))
@time resI2 = reduce(hcat,map(x->solve2(x,1f0,1f0,Float32.(getsim(4)),quad),X))
fig,ax = plt.subplots(1,3,figsize=(10,10))
ax[1].scatter(X[100:end],resH[1,100:end],color=(0,0,0),s=0.2)
ax[1].scatter(X[100:end],resH[2,100:end],color=(0,0,0),s=0.2)
ax[1].scatter(X[1:99],resH2[1,1:99],color=(0,0,0),s=0.2)
ax[1].scatter(X[1:99],resH2[2,1:99],color=(0,0,0),s=0.2)
ax[1].scatter(X[99:5:end],resH2[1,99:5:end],color=(0,0,0),s=0.2)
ax[1].scatter(X[99:5:end],resH2[2,99:5:end],color=(0,0,0),s=0.2)
ax[1].set_ylim(0,4)
ax[1].set_xlabel("inhibitory coupling")
ax[1].set_ylabel("rate")
ax[1].set_aspect("equal",adjustable="box")
ax[2].scatter(X[1:end],resK[1,1:end],color=(0,0,0),s=0.2)
ax[2].scatter(X[1:end],resK[2,1:end],color=(0,0,0),s=0.2)
ax[2].scatter(X[1:99],resK2[1,1:99],color=(0,0,0),s=0.2)
ax[2].scatter(X[1:99],resK2[2,1:99],color=(0,0,0),s=0.2)
ax[2].scatter(X[99:5:end],resK2[1,99:5:end],color=(0,0,0),s=0.2)
ax[2].scatter(X[99:5:end],resK2[2,99:5:end],color=(0,0,0),s=0.2)
ax[2].set_xlabel("inhibitory coupling")
ax[2].set_aspect("equal",adjustable="box")
ax[2].set_ylim(0,4)
ax[3].scatter(Ii[1:end],resI[1,1:end],color=(0,0,0),s=0.2)
ax[3].scatter(Ii[1:end],resI[2,1:end],color=(0,0,0),s=0.2)
ax[3].scatter(Ii[1:99],resI2[1,1:99],color=(0,0,0),s=0.2)
ax[3].scatter(Ii[1:99],resI2[2,1:99],color=(0,0,0),s=0.2)
ax[3].scatter(Ii[1:5:end],resI2[1,1:5:end],color=(0,0,0),s=0.2)
ax[3].scatter(Ii[1:5:end],resI2[2,1:5:end],color=(0,0,0),s=0.2)
ax[3].set_xlabel("External input Iₑ")
ax[3].set_ylabel("rate")
ax[3].set_aspect("equal",adjustable="box")
display(fig)
fig.savefig(joinpath(path,"figures","3.png"))


# Figure 4:
# we use a strategy where we integrate separately for each angles.
# so we don't use the simplified equations derived in the model.
# this is due to the fact that
Kconnect(K0,K1) = θ -> K0 + K1*cos(θ)
Hconnect(H0,H1) = θ -> H0 + H1*cos(θ)
θ_bin = Float32.(range(-π,stop=π,length=100))
dθ_bin = θ_bin[2]-θ_bin[1]
function system!(ds,s,p,t)
    IE,IL,IR,Ii,K0,K1,H0,H1,L0,α,ϕ = p
    se = s[1,:]
    sl = s[2,:]
    sr = s[3,:]
    Kc = Kconnect(K0,K1)
    Hc = Hconnect(H0,H1)
    ds[1,:] = -se./τ .+ ϕ.(IE(t) .- map(θ->sum(Kc.((θ-π+α).-θ_bin).*sl+Kc.((θ-π-α).-θ_bin).*sr)*dθ_bin/(4f0*π),θ_bin))./τ
    ds[2,:] = -sl./τ .+ ϕ.(Ii .+ IL(t) .+ map(θ->sum(Hc.(θ.-θ_bin).*se-L0.*sr)*dθ_bin/(2f0*π),θ_bin))./τ #error in this term in the paper.
    ds[3,:] = -sr./τ .+ ϕ.(Ii .+ IR(t) .+ map(θ->sum(Hc.(θ.-θ_bin).*se-L0.*sl)*dθ_bin/(2f0*π),θ_bin))./τ
end
function prob_func(prob,i,repeat)
    prob.u0 .= Float32.(zeros(3,size(θ_bin,1))) .+ 10f0
    k = K[i]
    L0 = 0.5f0
    H0 = 1.5f0
    H1 = 1.5f0
    α = π/3f0
    Ii = 0f0
    @. prob.p = [t->10f0*(1f0+k),t->0f0,t->0f0,Ii,k,k,H0,H1,L0,α,relu]
    prob
end
s0 = Float32.(rand(Uniform(0f0,1f0),(3,size(θ_bin,1))))
s0 .= 10f0
tspan = (0f0,1f0)
K = 0.0f0:0.5f0:20f0

#Note: we also tried the resolution using SteadyStateProblem instead of ODEProblem
# but the solver choosen automatically seemed not to converge properly,
# where the solver automatically used in ODEProblem converged well.

output_func(s,i) = (s.u[end],false)
p0 = [t->0f0,t->0f0,t->0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0,relu]
prob = EnsembleProblem(ODEProblem(system!,s0,tspan,p0),prob_func=prob_func,output_func=output_func)
@time sol = solve(prob;trajectories=size(K,1))
# The amount of time for this run was: 230s

#plot of the tuning curves (just for me)
fig,ax = plt.subplots(1,3,figsize=(10,10))
cmap=plt.get_cmap("jet")
for tpl in enumerate(sol.u)
    j,tc = tpl
    for i in 1:1:3
        ax[i].plot(θ_bin,tc[i,:],c=cmap(j/41))
    end
end
display(fig)

# plot of the maximal firing rate as a function of the coupling strength
u = reduce(hcat,map(s->maximum(s,dims=2),sol.u))
fig,ax = plt.subplots(2,1)
for s in collect(eachrow(u))
    ax[2].plot(K,s)
end
ax[2].set_ylabel("firing rate")
ax[2].set_ylim(0,50)
ax[2].set_xlabel("coupling strength K")
# plot of the tuning width as a function of the coupling strength
function get_width(tc)
    firstzero = 0f0 in tc ? argmax(-tc) : return 180
    ϕa = -π # it is clear from the tuning curve plot that the function reaches its maximum at -π
    θa = abs(ϕa - θ_bin[firstzero])*180/π
end
width = reduce(hcat,map(s->get_width.(collect(eachrow(s))),sol.u))
for s in collect(eachrow(width))
    ax[1].plot(K,s)
end
ax[1].set_ylabel("tuning width (deg)")
ax[1].set_xlabel("coupling strength K")
ax[1].set_ylim(60,210)
fig.tight_layout()
display(fig)
fig.savefig(joinpath(path,"figures","4.png"))
