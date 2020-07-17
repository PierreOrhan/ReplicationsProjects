## Idea:
#   We start with perfect connectivity of standard HD integrator model
# With this perfect connectivity we show that the inhibitory cells tuning remains very uniform through time.
# We then add a noise factor parametrized by a ϵ term which quantifies the scale of the perturbation.
#   We then run the network either using or not using STDP.
#
#   It would be interesting to see if we could write down the case without STDP,
#   and find a power spectrum of a matrix of interst, showing the possible degeneracy level and maybe quantifying their probability distribution.

# For the connectivity, it would make sense to use one based on previous paper



## REFLECTION AFTER MEETING WITH ADRIEN
# USE random connectivity input to small to large nb of FS cells
# The stable bump will stabilized by Martinotti cells.
# FS cells should be connected as a function of their anatomical distances.

# Similar effect than in figure 2B of the Brunel paper:
# Position ourself in the zone of mid to  high input where the transitioning is abrupt enough
#(in fact it could even not matter)
# Small underrepresentation in the stable bump of the HD pyr network will diminish the total input to FS cells
# leading to a switch to the 2nd stable state: bump emerge in the pop of inhibitory neurons
# The number of bump is function of the size of the population
#  (maybe smaller pop --> more bump (because of connectivity looping faster???))
#       Or larger pop: more chance to connect with a far distant neuron --> boom double frequency.

# The key is that these bumps will then act on the excitatory population back again, creating a sort of additional variation in their bump size
# How could the system revert back to its higher state ??
# The inhibitory act irrespectively of the angle on the excitatory population.

using Flux
using PyPlot
using DifferentialEquations
using Distributions

# rate networks experiments
# 2 inhibitory interneurons are interconnected and connected on a excitatory cell, which firing rate
τ_i = 0.01f0 # 1 ms --> the neurons firing rates converges very rapidly
τ_e = 0.01f0# 10 ms --> the neuron firing rates converges more slowly
IE = 10f0
Ii = 30f0

dt = 0.0001f0 #0.1 ms
T = 10 #10s
function solve2(Ie,K1,H1,L0,f0::Array{Float32,1}, ϕ = relu)
    f = zeros(Float32,(Int(div(T,dt)),3))
    f[1,:] =f0
    for idx in 1:1:size(f,1)-1
        #excitatory
        f[idx+1,1] = ϕ(IE+Ie-0.1f0*K1*(f[idx,2]+f[idx,3]))*dt/τ_e + f[idx,1]*(1-dt/τ_e)
        #inhibitory
        f[idx+1,2] = ϕ(Ii+Ie+H1*f[idx,1]-L0*f[idx,3])*dt/τ_i + f[idx,2]*(1-dt/τ_i)
        f[idx+1,3] = ϕ(Ii+Ie+H1*f[idx,1]-L0*f[idx,2])*dt/τ_i + f[idx,3]*(1-dt/τ_i)
    end
    return f[end,:]
end
X = 0f0:0.1f0:15f0
function getsim(nb=2)
    f = rand(Uniform(0f0,1f0),1)[1]
    return [f for _ in 1:1:nb]
end
quad(I) = I<0f0 ? 0f0 : I<=1f0 ? I*I : 2f0*√(I-3f0/4f0)

simuled_function = relu
@time resH = reduce(hcat,map(x->solve2(1f0,1f0,x,1f0,Float32.(rand(Uniform(0f0,1f0),3)),simuled_function),X))
@time resK = reduce(hcat,map(x->solve2(1f0,x,1f0,1f0,Float32.(rand(Uniform(0f0,1f0),3)),simuled_function),X))
@time resI = reduce(hcat,map(x->solve2(x,1f0,1f0,5f0,Float32.(rand(Uniform(0f0,1f0),3)),simuled_function),X))
@time resL = reduce(hcat,map(x->solve2(1f0,1f0,1f0,x,Float32.(rand(Uniform(0f0,1f0),3)),simuled_function),X))
@time resH2 = reduce(hcat,map(x->solve2(1f0,1f0,x,1f0,Float32.(getsim(3)),simuled_function),X))
@time resK2 = reduce(hcat,map(x->solve2(1f0,x,1f0,1f0,Float32.(getsim(3)),simuled_function),X))
@time resI2 = reduce(hcat,map(x->solve2(x,1f0,1f0,5f0,Float32.(getsim(3)),simuled_function),X))
@time resL2 = reduce(hcat,map(x->solve2(1f0,1f0,1f0,x,Float32.(getsim(3)),simuled_function),X))

sizeDot = 0.6
fig,ax = plt.subplots(2,2,figsize=(10,10))
ax[1,1].scatter(X[1:end],resH[1,1:end],color=(0,0,0),s=sizeDot)
ax[1,1].scatter(X[1:end],resH[2,1:end],color=(1,0,0),s=sizeDot)
ax[1,1].scatter(X[1:end],resH[3,1:end],color=(0,1,0),s=sizeDot)
ax[1,1].scatter(X[1:end],resH2[3,1:end],color=(0,0,1),s=sizeDot)
ax[1,1].scatter(X[1:end],resH2[1,1:end],color=(0,1,1),s=sizeDot)

ax[1,2].scatter(X[1:end],resK[1,1:end],color=(0,0,0),s=sizeDot)
ax[1,2].scatter(X[1:end],resK[2,1:end],color=(1,0,0),s=sizeDot)
ax[1,2].scatter(X[1:end],resK[3,1:end],color=(0,1,0),s=sizeDot)
ax[1,2].scatter(X[1:end],resK2[3,1:end],color=(0,0,1),s=sizeDot)
ax[1,2].scatter(X[1:end],resK2[1,1:end],color=(0,1,1),s=sizeDot)

ax[2,1].scatter(X[1:end],resI[1,1:end],color=(0,0,0),s=sizeDot)
ax[2,1].scatter(X[1:end],resI[2,1:end],color=(1,0,0),s=sizeDot)
ax[2,1].scatter(X[1:end],resI[3,1:end],color=(0,1,0),s=sizeDot)
ax[2,1].scatter(X[1:end],resI2[3,1:end],color=(0,0,1),s=sizeDot)
ax[2,1].scatter(X[1:end],resI2[1,1:end],color=(0,1,1),s=sizeDot)

ax[2,2].scatter(X[1:end],resL[1,1:end],color=(0,0,0),s=sizeDot)
ax[2,2].scatter(X[1:end],resL[2,1:end],color=(1,0,0),s=sizeDot)
ax[2,2].scatter(X[1:end],resL[3,1:end],color=(0,1,0),s=sizeDot)
ax[2,2].scatter(X[1:end],resL2[3,1:end],color=(0,0,1),s=sizeDot)
ax[2,2].scatter(X[1:end],resL2[1,1:end],color=(0,1,1),s=sizeDot)
display(fig)


# 2nd Experiment: we add another excitatory cells.
# our idea is that the system will be in one state when one excitatory is on
# and in another when the other one is on.

dt = 0.0001f0 #0.1 ms
T = 10 #10s
function solve2(Ie1,K1,H1,L0,f0::Array{Float32,1}, ϕ = relu)
    f = zeros(Float32,(Int(div(T,dt)),4))
    f[1,:] =f0
    for idx in 1:1:size(f,1)-1
        #excitatory
        f[idx+1,1] = ϕ(IE+Ie1-0.1f0*K1*(f[idx,3]+f[idx,4]))*dt/τ_e + f[idx,1]*(1-dt/τ_e)
        f[idx+1,2] = ϕ(IE+Ie1-0.1f0*K1*(f[idx,3]+f[idx,4]))*dt/τ_e + f[idx,2]*(1-dt/τ_e)
        #inhibitory
        f[idx+1,3] = ϕ(Ii+Ie1+H1*f[idx,1]+H1*f[idx,2]-L0*f[idx,4])*dt/τ_i + f[idx,3]*(1-dt/τ_i)
        f[idx+1,4] = ϕ(Ii+Ie1+H1*f[idx,1]+H1*f[idx,2]-L0*f[idx,3])*dt/τ_i + f[idx,4]*(1-dt/τ_i)
    end
    return f[end,:]
end
X = 0f0:0.1f0:15f0
function getsim(nb=2)
    f = rand(Uniform(0f0,1f0),1)[1]
    return [f for _ in 1:1:nb]
end
quad(I) = I<0f0 ? 0f0 : I<=1f0 ? I*I : 2f0*√(I-3f0/4f0)

simuled_function = relu

@time resH = reduce(hcat,map(x->solve2(1f0,1f0,x,5f0,Float32.(rand(Uniform(0f0,1f0),4)),simuled_function),X))
@time resK = reduce(hcat,map(x->solve2(1f0,x,1f0,5f0,Float32.(rand(Uniform(0f0,1f0),4)),simuled_function),X))
@time resI = reduce(hcat,map(x->solve2(x,1f0,1f0,5f0,Float32.(rand(Uniform(0f0,1f0),4)),simuled_function),X))
@time resL = reduce(hcat,map(x->solve2(1f0,1f0,1f0,x,Float32.(rand(Uniform(0f0,1f0),4)),simuled_function),X))
@time resH2 = reduce(hcat,map(x->solve2(1f0,1f0,x,5f0,Float32.(getsim(4)),simuled_function),X))
@time resK2 = reduce(hcat,map(x->solve2(1f0,x,1f0,5f0,Float32.(getsim(4)),simuled_function),X))
@time resI2 = reduce(hcat,map(x->solve2(x,1f0,1f0,5f0,Float32.(getsim(4)),simuled_function),X))
@time resL2 = reduce(hcat,map(x->solve2(1f0,1f0,1f0,x,Float32.(getsim(4)),simuled_function),X))

sizeDot = 0.6
fig,ax = plt.subplots(2,2,figsize=(10,10))
ax[1,1].scatter(X[1:end],resH[1,1:end],color=(0,0,0),s=sizeDot)
ax[1,1].scatter(X[1:end],resH[2,1:end],color=(0.5,0.5,0.5),s=sizeDot)
ax[1,1].scatter(X[1:end],resH[3,1:end],color=(1,0,0),s=sizeDot)
ax[1,1].scatter(X[1:end],resH[4,1:end],color=(0,1,0),s=sizeDot)
#case same initialization:
ax[1,1].scatter(X[1:end],resH2[4,1:end],color=(0,0,1),s=sizeDot)
ax[1,1].scatter(X[1:end],resH2[1,1:end],color=(0,1,1),s=sizeDot)

ax[1,2].scatter(X[1:end],resK[1,1:end],color=(0,0,0),s=sizeDot)
ax[1,2].scatter(X[1:end],resK[2,1:end],color=(0.5,0.5,0.5),s=sizeDot)
ax[1,2].scatter(X[1:end],resK[3,1:end],color=(1,0,0),s=sizeDot)
ax[1,2].scatter(X[1:end],resK[3,1:end],color=(0,1,0),s=sizeDot)
ax[1,2].scatter(X[1:end],resK2[3,1:end],color=(0,0,1),s=sizeDot)
ax[1,2].scatter(X[1:end],resK2[1,1:end],color=(0,1,1),s=sizeDot)

ax[2,1].scatter(X[1:end],resI[1,1:end],color=(0,0,0),s=sizeDot)
ax[2,1].scatter(X[1:end],resI[1,1:end],color=(0.5,0.5,0.5),s=sizeDot)
ax[2,1].scatter(X[1:end],resI[3,1:end],color=(1,0,0),s=sizeDot)
ax[2,1].scatter(X[1:end],resI[4,1:end],color=(0,1,0),s=sizeDot)
ax[2,1].scatter(X[1:end],resI2[4,1:end],color=(0,0,1),s=sizeDot)
ax[2,1].scatter(X[1:end],resI2[1,1:end],color=(0,1,1),s=sizeDot)

ax[2,2].scatter(X[1:end],resL[1,1:end],color=(0,0,0),s=sizeDot)
ax[2,2].scatter(X[1:end],resL[2,1:end],color=(0.5,0.5,0.5),s=sizeDot)
ax[2,2].scatter(X[1:end],resL[3,1:end],color=(1,0,0),s=sizeDot)
ax[2,2].scatter(X[1:end],resL[4,1:end],color=(0,1,0),s=sizeDot)
ax[2,2].scatter(X[1:end],resL2[4,1:end],color=(0,0,1),s=sizeDot)
ax[2,2].scatter(X[1:end],resL2[1,1:end],color=(0,1,1),s=sizeDot)
display(fig)


dt = 0.0001f0 #0.1 ms
T = 10 #10s
function solve2(Ie1,K1,H1,L0,f0::Array{Float32,1}, ϕ = relu)
    f = zeros(Float32,(Int(div(T,dt)),4))
    f[1,:] =f0
    for idx in 1:1:size(f,1)-1
        #excitatory
        f[idx+1,1] = ϕ(IE+Ie1-0.1f0*K1*(f[idx,3]+f[idx,4]))*dt/τ_e + f[idx,1]*(1-dt/τ_e)
        #inhibitory
        f[idx+1,2] = ϕ(Ii+Ie1+H1*f[idx,1]-L0*f[idx,4])*dt/τ_i + f[idx,2]*(1-dt/τ_i)
        f[idx+1,3] = ϕ(Ii+Ie1+H1*f[idx,1]-L0*f[idx,2])*dt/τ_i + f[idx,3]*(1-dt/τ_i)
        f[idx+1,4] = ϕ(Ii+Ie1+H1*f[idx,1]-L0*f[idx,3])*dt/τ_i + f[idx,4]*(1-dt/τ_i)
    end
    return f[end,:]
end
X = 0f0:0.1f0:15f0
function getsim(nb=2)
    f = rand(Uniform(0f0,1f0),1)[1]
    return [f for _ in 1:1:nb]
end
quad(I) = I<0f0 ? 0f0 : I<=1f0 ? I*I : 2f0*√(I-3f0/4f0)

simuled_function = relu

@time resH = reduce(hcat,map(x->solve2(1f0,1f0,x,5f0,Float32.(rand(Uniform(0f0,1f0),4)),simuled_function),X))
@time resK = reduce(hcat,map(x->solve2(1f0,x,1f0,5f0,Float32.(rand(Uniform(0f0,1f0),4)),simuled_function),X))
@time resI = reduce(hcat,map(x->solve2(x,1f0,1f0,5f0,Float32.(rand(Uniform(0f0,1f0),4)),simuled_function),X))
@time resL = reduce(hcat,map(x->solve2(1f0,1f0,1f0,x,Float32.(rand(Uniform(0f0,1f0),4)),simuled_function),X))
@time resH2 = reduce(hcat,map(x->solve2(1f0,1f0,x,5f0,Float32.(getsim(4)),simuled_function),X))
@time resK2 = reduce(hcat,map(x->solve2(1f0,x,1f0,5f0,Float32.(getsim(4)),simuled_function),X))
@time resI2 = reduce(hcat,map(x->solve2(x,1f0,1f0,5f0,Float32.(getsim(4)),simuled_function),X))
@time resL2 = reduce(hcat,map(x->solve2(1f0,1f0,1f0,x,Float32.(getsim(4)),simuled_function),X))

sizeDot = 0.6
fig,ax = plt.subplots(2,2,figsize=(10,10))
ax[1,1].scatter(X[1:end],resH[1,1:end],color=(0,0,0),s=sizeDot)
ax[1,1].scatter(X[1:end],resH[2,1:end],color=(0.5,0.5,0.5),s=sizeDot)
ax[1,1].scatter(X[1:end],resH[3,1:end],color=(1,0,0),s=sizeDot)
ax[1,1].scatter(X[1:end],resH[4,1:end],color=(0,1,0),s=sizeDot)
#case same initialization:
ax[1,1].scatter(X[1:end],resH2[4,1:end],color=(0,0,1),s=sizeDot)
ax[1,1].scatter(X[1:end],resH2[1,1:end],color=(0,1,1),s=sizeDot)

ax[1,2].scatter(X[1:end],resK[1,1:end],color=(0,0,0),s=sizeDot)
ax[1,2].scatter(X[1:end],resK[2,1:end],color=(0.5,0.5,0.5),s=sizeDot)
ax[1,2].scatter(X[1:end],resK[3,1:end],color=(1,0,0),s=sizeDot)
ax[1,2].scatter(X[1:end],resK[3,1:end],color=(0,1,0),s=sizeDot)
ax[1,2].scatter(X[1:end],resK2[3,1:end],color=(0,0,1),s=sizeDot)
ax[1,2].scatter(X[1:end],resK2[1,1:end],color=(0,1,1),s=sizeDot)

ax[2,1].scatter(X[1:end],resI[1,1:end],color=(0,0,0),s=sizeDot)
ax[2,1].scatter(X[1:end],resI[1,1:end],color=(0.5,0.5,0.5),s=sizeDot)
ax[2,1].scatter(X[1:end],resI[3,1:end],color=(1,0,0),s=sizeDot)
ax[2,1].scatter(X[1:end],resI[4,1:end],color=(0,1,0),s=sizeDot)
ax[2,1].scatter(X[1:end],resI2[4,1:end],color=(0,0,1),s=sizeDot)
ax[2,1].scatter(X[1:end],resI2[1,1:end],color=(0,1,1),s=sizeDot)

ax[2,2].scatter(X[1:end],resL[1,1:end],color=(0,0,0),s=sizeDot)
ax[2,2].scatter(X[1:end],resL[2,1:end],color=(0.5,0.5,0.5),s=sizeDot)
ax[2,2].scatter(X[1:end],resL[3,1:end],color=(1,0,0),s=sizeDot)
ax[2,2].scatter(X[1:end],resL[4,1:end],color=(0,1,0),s=sizeDot)
ax[2,2].scatter(X[1:end],resL2[4,1:end],color=(0,0,1),s=sizeDot)
ax[2,2].scatter(X[1:end],resL2[1,1:end],color=(0,1,1),s=sizeDot)
display(fig)



dt = 0.0001f0 #0.1 ms
T = 10 #10s
τ_a = 0.01f0
τ_i = 0.001f0
τ_e = 0.01f0
A = 1f0
function solve2(Ie1,K1,H1,L0,L1,L2,f0::Array{Float32,1}, ϕ = relu)
    f = zeros(Float32,(Int(div(T,dt)),3))
    f[1,:] =f0
    for idx in 1:1:size(f,1)-1
        #excitatory:
        f[idx+1,1] = ϕ(Ie1-L1*f[idx,2]-L2*f[idx,3])*dt/τ_e + f[idx,1]*(1-dt/τ_e)
        #FS inhibitory:
        f[idx+1,2] = ϕ(Ie1+60f0+H1*f[idx,1]-L0*f[idx,2])*dt/τ_i + f[idx,2]*(1-dt/τ_i)
        #NFC (MC cells) inhibitory:
        f[idx+1,3] = ϕ(f[idx,1]*A-L0*f[idx,3])*dt/τ_a + f[idx,3]*(1-dt/τ_a)
    end
    return f[end,:]
end
simuled_function = relu
X = 0f0:0.2f0:15f0
Ie1,K1,H1,L0,L1,L2 = 20f0,1f0,1f0,1f0,0.1f0,1f0
@time resH = reduce(hcat,map(x->solve2(Ie1,x,H1,L0,L1,L2,Float32.(rand(Uniform(0f0,1f0),3)),simuled_function),X))
sizeDot = 3
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(X[1:end],resH[1,1:end],color=(0,0,0),s=sizeDot)
ax.scatter(X[1:end],resH[2,1:end],color=(0.5,0.5,0.5),s=sizeDot)
ax.scatter(X[1:end],resH[3,1:end],color=(1,0,0),s=sizeDot)
display(fig)

fig,ax = plt.subplots(1,1)
ax.plot(resL[80000:end,1])
ax.plot(resL[80000:end,2])
ax.plot(resL[80000:end,3])
ax.set_ylim(0,100)
display(fig)



dt = 0.0001f0 #0.1 ms
T = 1 #10s
τ_i = 0.001f0
function solve2(Ie1,L0,f0::Array{Float32,1}, ϕ = relu)
    f = zeros(Float32,(Int(div(T,dt)),2))
    f[1,:] =f0
    noise_i1 = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    noise_i2 = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    for idx in 1:1:size(f,1)-1
        #FS inhibitory1:
        f[idx+1,1] = ϕ(noise_i1[idx]+Ie1-L0*f[idx,2])*dt/τ_i + f[idx,1]*(1-dt/τ_i)
        #FS inhibitory2:
        f[idx+1,2] = ϕ(noise_i2[idx]+Ie1-L0*f[idx,1])*dt/τ_i + f[idx,2]*(1-dt/τ_i)
    end
    return f[end,:]
end
simuled_function = quad
Ie1,L0 = 3.8f0,1f0
XIe = 1.4f0:0.05f0:6f0
@time resIe = reduce(hcat,map(x->solve2(x,1.0f0,Float32.(rand(Uniform(0f0,1f0),2)),quad),XIe))
XL0 = 0.5f0:0.05f0:6f0
@time resL = reduce(hcat,map(x->solve2(Ie1,x,Float32.(rand(Uniform(0f0,1f0),2)),quad),XL0))
sizeDot = 3
fig,ax = plt.subplots(2,2,figsize=(10,10))
ax[1,1].scatter(XIe[1:end],resIe[1,1:end],color=(0,0,1),s=sizeDot)
ax[1,1].scatter(XIe[1:end],resIe[2,1:end],color=(1,0,0),s=sizeDot)
ax[2,1].scatter(XIe[1:end],resIe[2,1:end].+resIe[1,:],color=(0,0,0),s=sizeDot)
ax[1,2].scatter(XL0[1:end],resL[1,1:end],color=(0,0,1),s=sizeDot)
ax[1,2].scatter(XL0[1:end],resL[2,1:end],color=(1,0,0),s=sizeDot)
ax[2,2].scatter(XL0[1:end],resL[2,1:end].+resL[2,:],color=(0,0,0),s=sizeDot)
display(fig)

dt = 0.0001f0 #0.1 ms
T = 1 #10s
τ_i = 0.01f0
τ_e = 0.01f0
IE = 3f0
Ie1 = 3.8f0
L0 = 1f0
L1= 1f0
function solve_2pop(Ie1,IE,L0,L1,f0::Array{Float32,1}, ϕ = relu)
    f = zeros(Float32,(Int(div(T,dt)),3))
    f[1,:] =f0
    noise_e = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    noise_i1 = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    noise_i2 = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    for idx in 1:1:size(f,1)-1
        #excitatory cells are solely loosely modulated by the FS inhibitory network
        # so L1 <<< L0
        f[idx+1,1] = ϕ(noise_e[idx]+Ie1+IE-L1*(f[idx,2]+f[idx,3]))*dt/τ_e + f[idx,1]*(1-dt/τ_e)
        #FS inhibitory1:
        f[idx+1,2] = ϕ(noise_i1[idx]+Ie1-L0*f[idx,3]+0.1f0*f[idx,1])*dt/τ_i + f[idx,2]*(1-dt/τ_i)
        #FS inhibitory2: #+
        f[idx+1,3] = ϕ(noise_i2[idx]+Ie1-L0*f[idx,2]+0.1f0*f[idx,1])*dt/τ_i + f[idx,3]*(1-dt/τ_i)
    end
    return f[end,:]
end
# we need to find constants such that
XInput = 0f0:0.05f0:10f0
XL0= 0f0:0.05f0:10f0
@time resIe = reduce(hcat,map(x->solve_2pop(x,IE,L0,L1,Float32.(rand(Uniform(0f0,1f0),3)),quad),XInput))
@time resL = reduce(hcat,map(x->solve_2pop(3,x,L0,L1,Float32.(rand(Uniform(0f0,1f0),3)),quad),XL0))
fig,ax = plt.subplots(2,2,figsize=(10,10))
ax[1,1].scatter(XInput[1:end],resIe[1,1:end],color=(0,0,1),s=sizeDot)
ax[1,1].scatter(XInput[1:end],resIe[2,1:end],color=(1,0,0),s=sizeDot)
ax[1,1].scatter(XInput[1:end],resIe[3,1:end],color=(0,1,0),s=sizeDot)
ax[2,1].scatter(XInput[1:end],resIe[2,1:end].+resIe[3,:],color=(0,0,0),s=sizeDot)
ax[1,2].scatter(XL0[1:end],resL[1,1:end],color=(0,0,1),s=sizeDot)
ax[1,2].scatter(XL0[1:end],resL[2,1:end],color=(1,0,0),s=sizeDot)
ax[1,2].scatter(XL0[1:end],resL[3,1:end],color=(0,1,0),s=sizeDot)
ax[2,2].scatter(XL0[1:end],resL[2,1:end].+resL[3,:],color=(0,0,0),s=sizeDot)
display(fig)



dt = 0.0001f0 #0.1 ms
T = 1 #10s
τ_i = 0.01f0
τ_e = 0.01f0
IE = 3f0
Ie1 = 3.8f0
L0 = 1f0
L1= 1f0
function solve_2pop(Ie1,IE,L0,L1,f0::Array{Float32,1}, ϕ = relu)
    f = zeros(Float32,(Int(div(T,dt)),3))
    f[1,:] =f0
    noise_e = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    noise_i1 = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    noise_i2 = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    for idx in 1:1:size(f,1)-1
        #excitatory cells are solely loosely modulated by the FS inhibitory network
        # so L1 <<< L0
        f[idx+1,1] = ϕ(noise_e[idx]+Ie1+IE-L1*(f[idx,2]+f[idx,3]))*dt/τ_e + f[idx,1]*(1-dt/τ_e)
        #FS inhibitory1:
        f[idx+1,2] = ϕ(noise_i1[idx]+Ie1-L0*f[idx,3]+0.1f0*f[idx,1])*dt/τ_i + f[idx,2]*(1-dt/τ_i)
        #FS inhibitory2: #+
        f[idx+1,3] = ϕ(noise_i2[idx]+Ie1-L0*f[idx,2]+0.1f0*f[idx,1])*dt/τ_i + f[idx,3]*(1-dt/τ_i)
    end
    return f[end,:]
end
# we need to find constants such that
XInput = 0f0:0.05f0:10f0
XL0= 0f0:0.05f0:10f0
@time resIe = reduce(hcat,map(x->solve_2pop(x,IE,L0,L1,Float32.(rand(Uniform(0f0,1f0),3)),quad),XInput))
@time resL = reduce(hcat,map(x->solve_2pop(Ie1,x,L0,L1,Float32.(rand(Uniform(0f0,1f0),3)),quad),XL0))
fig,ax = plt.subplots(2,2,figsize=(10,10))
ax[1,1].scatter(XInput[1:end],resIe[1,1:end],color=(0,0,1),s=sizeDot)
ax[1,1].scatter(XInput[1:end],resIe[2,1:end],color=(1,0,0),s=sizeDot)
ax[1,1].scatter(XInput[1:end],resIe[3,1:end],color=(0,1,0),s=sizeDot)
ax[2,1].scatter(XInput[1:end],resIe[2,1:end].+resIe[3,:],color=(0,0,0),s=sizeDot)
ax[1,2].scatter(XL0[1:end],resL[1,1:end],color=(0,0,1),s=sizeDot)
ax[1,2].scatter(XL0[1:end],resL[2,1:end],color=(1,0,0),s=sizeDot)
ax[1,2].scatter(XL0[1:end],resL[3,1:end],color=(0,1,0),s=sizeDot)
ax[2,2].scatter(XL0[1:end],resL[2,1:end].+resL[3,:],color=(0,0,0),s=sizeDot)
display(fig)


# 3D plots

dt = 0.0001f0 #0.1 ms
T = 1 #10s
τ_i = 0.01f0
τ_e = 0.01f0
IE = 3f0
Ie1 = 3.8f0
L0 = 1f0
L1= 1f0
function solve_2pop(Ie1,IE,L0,L1,f0::Array{Float32,1}, ϕ = relu)
    f = zeros(Float32,(Int(div(T,dt)),3))
    f[1,:] =f0
    noise_e = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    noise_i1 = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    noise_i2 = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    for idx in 1:1:size(f,1)-1
        #excitatory cells are solely loosely modulated by the FS inhibitory network
        # so L1 <<< L0
        f[idx+1,1] = ϕ(noise_e[idx]+Ie1+IE-L1*(f[idx,2]+f[idx,3]))*dt/τ_e + f[idx,1]*(1-dt/τ_e)
        #FS inhibitory1:
        f[idx+1,2] = ϕ(noise_i1[idx]+Ie1-L0*f[idx,3]+0.1f0*f[idx,1])*dt/τ_i + f[idx,2]*(1-dt/τ_i)
        #FS inhibitory2: #+
        f[idx+1,3] = ϕ(noise_i2[idx]+Ie1-L0*f[idx,2]+0.1f0*f[idx,1])*dt/τ_i + f[idx,3]*(1-dt/τ_i)
    end
    return f[end,:]
end
# we need to find constants such that
XInput = 0f0:0.05f0:2f0
XL0= 0f0:0.05f0:5f0
@time res3D = map(y->reduce(hcat,map(x->solve_2pop(x,IE,y,L1,Float32.(rand(Uniform(0f0,1f0),3)),quad),XInput)),XL0)
using3D()
fig = plt.figure()
ax1 = fig.add_subplot(111,projection="3d")
fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection="3d")
fig3 = plt.figure()
ax3 = fig3.add_subplot(111,projection="3d")
for tpl in enumerate(XL0)
    idx,l0= tpl
    r =res3D[idx]
    ax1.plot3D(XInput,r[1,:],l0)
    ax1.set_zlabel("L0")
    ax2.scatter(XInput,l0.+zeros(size(r,2)),r[2,:],s=sizeDot,c=[(0,0,1) for _ in 1:1:size(r,2)])
    ax2.set_xlabel("Xinput")
    ax3.scatter(l0.+zeros(size(r,2)),XInput,r[3,:],s=sizeDot,c=[(1,0,0) for _ in 1:1:size(r,2)])
end
display(fig)
display(fig2)
display(fig3)


dt = 0.0001f0 #0.1 ms
T = 1 #10s
τ_i = 0.01f0
τ_e = 0.01f0
IE = 3f0
Ie1 = 3.8f0
L0 = 1f0
L1= 1f0
function solve_2pop(θ,IE,L0,L1,f0::Array{Float32,1}, ϕ = relu)
    f = zeros(Float32,(Int(div(T,dt)),4))
    f[1,:] =f0
    noise_e = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    noise_i1 = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    noise_i2 = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    noise_i3 = Float32.(rand(Uniform(0,0.1),Int(div(T,dt))))
    for idx in 1:1:size(f,1)-1
        #excitatory cells are solely loosely modulated by the FS inhibitory network
        # so L1 <<< L0
        f[idx+1,1] = ϕ(noise_e[idx]+IE-L1*(f[idx,2]+f[idx,3]+f[idx,4]))*dt/τ_e + f[idx,1]*(1-dt/τ_e)
        #FS inhibitory1:
        f[idx+1,2] = ϕ(noise_i1[idx]+Ie1*cos(θ)-L0*f[idx,3]-L0*f[idx,4]+0.1f0*f[idx,1])*dt/τ_i + f[idx,2]*(1-dt/τ_i)
        #FS inhibitory2:
        f[idx+1,3] = ϕ(noise_i2[idx]+Ie1-L0*f[idx,2]-L0*f[idx,4]+0.1f0*f[idx,1])*dt/τ_i + f[idx,3]*(1-dt/τ_i)
        #FS inhibitory3:
        f[idx+1,4] = ϕ(noise_i3[idx]+Ie1-L0*f[idx,2]-L0*f[idx,3]+0.1f0*f[idx,1])*dt/τ_i + f[idx,4]*(1-dt/τ_i)
    end
    return f[end,:]
end
# we need to find constants such that
XInput = 0f0:0.05f0:10f0
XL0= 0f0:0.05f0:10f0
@time resIe = reduce(hcat,map(x->solve_2pop(x,IE,L0-0.5f0,L1,Float32.(rand(Uniform(0f0,1f0),4)),relu),XInput))
@time resL = reduce(hcat,map(x->solve_2pop(x,IE,L0,L1,Float32.(rand(Uniform(0f0,1f0),4)),relu),XInput))
@time resL2 = reduce(hcat,map(x->solve_2pop(x,IE,L0+0.5f0,L1,Float32.(rand(Uniform(0f0,1f0),4)),relu),XInput))
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax[1,1].scatter(XInput[1:end],resIe[1,1:end],color=(0,0,1),s=sizeDot)
ax[1,1].scatter(XInput[1:end],resIe[2,1:end],color=(1,0,0),s=sizeDot)
ax[1,1].scatter(XInput[1:end],resIe[3,1:end],color=(0,1,0),s=sizeDot)
ax[1,1].scatter(XInput[1:end],resIe[4,1:end],color=(1,1,0),s=sizeDot)
ax[2,1].scatter(XInput[1:end],resIe[2,1:end].+resIe[3,:].+resIe[4,:],color=(0,0,0),s=sizeDot)
ax[1,2].scatter(XL0[1:end],resL[1,1:end],color=(0,0,1),s=sizeDot)
ax[1,2].scatter(XL0[1:end],resL[2,1:end],color=(1,0,0),s=sizeDot)
ax[1,2].scatter(XL0[1:end],resL[3,1:end],color=(0,1,0),s=sizeDot)
ax[1,2].scatter(XInput[1:end],resL[4,1:end],color=(1,1,0),s=sizeDot)
ax[2,2].scatter(XL0[1:end],resL[2,1:end].+resL[3,:].+resL[4,:],color=(0,0,0),s=sizeDot)
ax[1,3].scatter(XL0[1:end],resL2[1,1:end],color=(0,0,1),s=sizeDot)
ax[1,3].scatter(XL0[1:end],resL2[2,1:end],color=(1,0,0),s=sizeDot)
ax[1,3].scatter(XL0[1:end],resL2[3,1:end],color=(0,1,0),s=sizeDot)
ax[2,3].scatter(XL0[1:end],resL2[2,1:end].+resL2[3,:].+resL2[4,:],color=(0,0,0),s=sizeDot)
ax[1,3].scatter(XInput[1:end],resL2[4,1:end],color=(1,1,0),s=sizeDot)
display(fig)





# Current experiments:
# First: we need to make sure we have the good modelisation for the FS cells.
# Experiment 1:
#   For a pyr pop control by a moving bump, all-to-all coupled to the inhibitory population
#   can we make a bump (or multiple bump appear in the inhibitory population)
#       Leading to 2 states of equilibrium for the dynamics?


# Experiment 2:
#  Change the number of inhibitory and excitatory ---> How is influenced the number of bump ?
#    My idea: larger network ==> more bump arise but they move at the same speed (only function of the recurrent weight)
#       therefore, the network will show more or less drastic path in the dynamic plane
#       Coupled with the excitatory pop : pattern emergence in the size of the bump!! --> 1/2/3 folds ect....
