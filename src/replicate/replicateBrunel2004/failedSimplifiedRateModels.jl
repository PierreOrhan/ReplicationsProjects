using PyPlot
using Flux
using LinearAlgebra
using BenchmarkTools
using ProgressBars
using Distributions
using Roots
using ForwardDiff
using DifferentialEquations

# 🔥 Exploratory code for figure 4 that failed
# for figure 4 the convergence is not easy to obtain and some of our strategy failed,
# here exposed.
# Notably the use of the extended system of equations is failing.
# While we first encountered some numerical issues, after being solved the problem persisted.

# stationary solutions for the ring attractor model
# We need to solve f(X) = X with f define as:
using DifferentialEquations
function system!(ds,s,p,t)
    Ie,Ii,K1,H0,L0,ϕ = p
    #excitatory
    ds[1] = ϕ(Ie-K1*s[2])/τ - s[1]/τ
    #inhibitory
    ds[2] = ϕ(Ii+H0*s[1])/(1f0+L0)/τ - s[2]/τ
end
L0 = 0.5f0
H0 = 1.5f0
H1 = 1.5f0
α = π/3f0
Ii = 0f0
s0 = [5.0f0,5.0f0]
tspan = (0f0,100f0)
k = 5.1f0
K = 0.1f0:0.5f0:20f0
p = [10f0*(1f0+k),Ii,k,H0,L0,relu]
#next we define a prob_func that modifies the prob by changing the K parameter:
function prob_func(prob,i,repeat)
    prob.u0 .= Float32.(rand(Uniform(0f0,1f0),2))
    k = K[i]
    @. prob.p = [10f0*(1f0+k),Ii,k,H0,L0,relu]
    prob
end
output_func(s,i) = (s.u[end],false)
prob = EnsembleProblem(ODEProblem(system!,s0,tspan,p),prob_func=prob_func,output_func=output_func)
@time sol = solve(prob;trajectories=size(K,1))
res  = reduce(hcat,sol.u)
fig,ax = plt.subplots()
ax.plot(K,res[1,:],linestyle="-",linewidth=3)
ax.plot(K,res[2,:],linestyle="--",linewidth=3)
ax.set_ylim(0f0,50f0)
display(fig)


function systemTot!(out,ds,s,p,t)
    IE,IL,IR,Ii,K0,K1,H0,H1,L0,α,ϕ = p
    se0,sl0,sr0,se1,sr1,sl1,ψe,ψl,ψr,ϕe = s
    #computes IA0 and IA1 terms A=e,r,l
    Ie0 = IE(t) - K0/2f0*(sl0+sr0)
    Il0 = Ii + IL(t) + H0*se0-L0*sr0
    Ir0 = Ii + IR(t) + H0*se0-L0*sl0
    Ie1 = K1/2f0*(cos(ψl-α-ϕe)*sl1 + cos(ψr+α-ϕe)*sr1)
    Il1 = H1*se1
    Ir1 = H1*se1
    #deduces the angle at which the firing rate profiles becomes 0:
    # the relu linearity is implemented using max(min(1f0,-Ie0/Ie1),-1f0))
    θe = acos(max(min(1f0,Ie0==0.0 ? 0.0 : -Ie0/Ie1),-1f0))
    θl = acos(max(min(1f0,Il0==0.0 ? 0.0 : -Il0/Il1),-1f0))
    θr = acos(max(min(1f0,Ir0==0.0 ? 0.0 : -Ir0/Ir1),-1f0))
    println(Ie0)
    println(" ",sl0," ",sr0)
    @assert !isnan(θe)
    @assert !isnan(θl)
    @assert !isnan(θr)
    #obtain the derivative.    #MEAN COMPONENTS
    out[1] = -ds[1] -se0/τ + (Ie0*θe/π+Ie1*sin(θe)/π)/τ
    out[2] = -ds[2] -sl0/τ + (Il0*θl/π+Il1*sin(θl)/π)/τ
    out[3] = -ds[3] -sr0/τ + (Ir0*θr/π+Ir1*sin(θr)/π)/τ
    #1st fourier components:
    out[4] = -ds[4] -se1/τ + (Ie0*sin(θe)/(2f0π)+Ie1*θe/(2f0*π)+Ie1*sin(2*θe)/(4f0*π))cos(ϕe-ψe)/τ
    out[5] = -ds[5] -sl1/τ + (Il0*sin(θl)/(2f0π)+Il1*θl/(2f0*π)+Il1*sin(2*θl)/(4f0*π))cos(ψe-ψl)/τ
    out[6] = -ds[6] -sr1/τ + (Ir0*sin(θr)/(2f0π)+Ir1*θr/(2f0*π)+Ir1*sin(2*θr)/(4f0*π))cos(ψe-ψr)/τ
    # updates of the ψA:
    out[7] = -ds[7]*se1 + (Ie0*sin(θe)/(2f0π)+Ie1*θe/(2f0*π)+Ie1*sin(2*θe)/(4f0*π))sin(ϕe-ψe)/τ
    out[8] = -ds[8]*sl1 +(Il0*sin(θl)/(2f0π)+Il1*θl/(2f0*π)+Il1*sin(2*θl)/(4f0*π))sin(ψe-ψl)/τ
    out[9] = -ds[9]*sr1 + (Ir0*sin(θr)/(2f0π)+Ir1*θr/(2f0*π)+Ir1*sin(2*θr)/(4f0*π))sin(ψe-ψr)/τ
    #Finally the equation that defines ϕe:
    out[10] = (sin(ψl-α-ϕe)*sl1 + sin(ψr+α-ϕe)*sr1)
    println(out)
end
tspan = (0.0,1.0)
K = 0.1:0.5:20
function prob_func(prob,i,repeat)
    prob.u0 .= zeros(10)
    prob.u0[1:3] .= 10.0
    #prob.u0 .= zeros(Float32,9)
    #prob.u0[1:3] .= 0.1f0 #set the initial ψ at 0f0
    k = K[i]
    L0 = 0.5
    H0 = 1.5
    H1 = 1.5
    α = π/3
    Ii = 0.0
    @. prob.p = [t->10.0*(1.0+k),t->0.0,t->0.0,Ii,k,k,H0,H1,L0,α,relu]
    prob
end
output_func(s,i) = (s.u[end],false)
differential_vars = [true,true,true,true,true,true,true,true,true,false]
p0 = [t->0.0,t->0.0,t->0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,relu]
s0 = rand(Uniform(0f0,1f0),10)
s0[1:3] .= 10
ds0 = rand(Uniform(0.0,1.0),10)
prob = DAEProblem(systemTot!,ds0,s0,tspan,p0,differential_vars=differential_vars)
prob_func(prob,1,false)
@time sol =  solve(prob)

prob = EnsembleProblem(DAEProblem(systemTot!,ds0,s0,tspan,p0,differential_vars=differential_vars),prob_func=prob_func,output_func=output_func)
@time sol = solve(prob;trajectories=size(K,1))
res  = reduce(hcat,sol.u)
fig,ax = plt.subplots()
ax.plot(K,res[1,:],linestyle="-",linewidth=3)
ax.plot(K,res[2,:],linestyle="--",linewidth=3)
ax.set_ylim(0f0,200f0)
display(fig)


## Next: strategy which search for the good ϕe at each time step:
#at first I tried an approach using Newton:
D(f) = x -> ForwardDiff.derivative(f,Float32(x))
zϕe(sl1,sr1,ψl,ψr) = ϕe -> (sin(ψl-α-ϕe)*sl1 + sin(ψr+α-ϕe)*sr1)
# # at each step we need to find ϕe according to the equation (sin(ψl-α-ϕe)*sl1 + sin(ψr+α-ϕe)*sr1) = 0
# println(ϕe)
#let us quit the perfect uniform state:
function f!(ds,s,p,t)
    IE,IL,IR,Ii,K0,K1,H0,H1,L0,α,ϕ = p
    se0,sl0,sr0,se1,sr1,sl1,ψe,ψl,ψr = s
    ##computes the angle which should represent the instantaneous location of the peak firing rate of the neuronal population:
    z = zϕe(sl1,sr1,ψl,ψr)
    ϕe =0f0
    try
        ϕe = find_zero(z,(0f0,π))
    catch
        ϕe = find_zero(z,(0.2f0,π+0.2f0))
    end
    #computes IA0 and IA1 terms A=e,r,l
    Ie0 = IE(t) - K0/2f0*(sl0+sr0)
    Il0 = Ii + IL(t) + H0*se0-L0*sr0
    Ir0 = Ii + IR(t) + H0*se0-L0*sl0
    Ie1 = K1/2f0*(cos(ψl-α-ϕe)*sl1 + cos(ψr+α-ϕe)*sr1)
    Il1 = H1*se1
    Ir1 = H1*se1
    #deduces the angle at which the firing rate profiles becomes 0:
    # the relu linearity is implemented using max(min(1f0,-Ie0/Ie1),-1f0))
    θe = acos(max(min(1f0,Ie0==0.0 ? 0.0 : -Ie0/Ie1),-1f0))
    θl = acos(max(min(1f0,Il0==0.0 ? 0.0 : -Il0/Il1),-1f0))
    θr = acos(max(min(1f0,Ir0==0.0 ? 0.0 : -Ir0/Ir1),-1f0))
    @assert !isnan(θe)
    @assert !isnan(θl)
    @assert !isnan(θr)
    println("s: ",s)
    println("ds: ",ds)
    #obtain the derivative.    #MEAN COMPONENTS
    ds[1] = -se0/τ + (Ie0*θe/π+Ie1*sin(θe)/π)/τ
    ds[2] = -sl0/τ + (Il0*θl/π+Il1*sin(θl)/π)/τ
    ds[3] = -sr0/τ + (Ir0*θr/π+Ir1*sin(θr)/π)/τ
    #1st fourier components:
    ds[4] = -se1/τ + (Ie0*sin(θe)/(2f0π)+Ie1*θe/(2f0*π)+Ie1*sin(2*θe)/(4f0*π))cos(ϕe-ψe)/τ
    ds[5] = -sl1/τ + (Il0*sin(θl)/(2f0π)+Il1*θl/(2f0*π)+Il1*sin(2*θl)/(4f0*π))cos(ψe-ψl)/τ
    ds[6] = -sr1/τ + (Ir0*sin(θr)/(2f0π)+Ir1*θr/(2f0*π)+Ir1*sin(2*θr)/(4f0*π))cos(ψe-ψr)/τ
    # updates of the ψA:
    ds[7] = (Ie0/se1*sin(θe)/(2f0π)+H1*θe/(2f0*π)+H1*sin(2*θe)/(4f0*π))sin(ϕe-ψe)/τ
    ds[8] = (Il0/sl1*sin(θl)/(2f0π)+H1*θl/(2f0*π)+H1*sin(2*θl)/(4f0*π))sin(ψe-ψl)/τ
    ds[9] = (Ir0/sr1*sin(θr)/(2f0π)+H1*θr/(2f0*π)+H1*sin(2*θr)/(4f0*π))sin(ψe-ψr)/τ
    println("new ds: ",ds)
end
tspan = (0.0,1.0)
K = 0.1:0.5:20
function prob_func(prob,i,repeat)
    prob.u0 .= Float32.(zeros(9))
    prob.u0[1:3] .= 10.0f0
    #prob.u0 .= zeros(Float32,9)
    #prob.u0[1:3] .= 0.1f0 #set the initial ψ at 0f0
    k = K[i]
    L0 = 0.5f0
    H0 = 1.5f0
    H1 = 1.5f0
    α = π/3.0f0
    Ii = 0.0f0
    @. prob.p = [t->10.0f0*(1.0f0+k),t->0.0f0,t->0.0f0,Ii,k,k,H0,H1,L0,α,relu]
    prob
end
output_func(s,i) = (s.u[end],false)
differential_vars = [true,true,true,true,true,true,true,true,true,false]
p0 = [t->0f0,t->0f0,t->0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0,relu]
s0 = Float32.(rand(Uniform(0f0,1f0),9))
prob = SteadyStateProblem(f!,s0,p0)
prob_func(prob,1,false)
@time sol =  solve(prob)



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
output_func(s,i) = (reshape(s.u,size(s0)),false)
p0 = [t->0f0,t->0f0,t->0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0,relu]
prob = EnsembleProblem(SteadyStateProblem(system!,s0,p0),prob_func=prob_func,output_func=output_func)
@time sol = solve(prob;trajectories=size(K,1))
fig,ax = plt.subplots(1,3)
cmap=plt.get_cmap("jet")
for tpl in enumerate(sol.u)
    j,tc = tpl
    for i in 1:1:3
        ax[i].plot(θ_bin,tc[i,:],c=cmap(j/41))
    end
end
display(fig)

u = reduce(hcat,map(s->maximum(s,dims=2),sol.u))
fig,ax = plt.subplots()
for s in collect(eachrow(u))
    ax.plot(K,s)
end
display(fig)
