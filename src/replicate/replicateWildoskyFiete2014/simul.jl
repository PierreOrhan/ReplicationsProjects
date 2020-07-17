#simulation of the training on the linear track
using CuArrays
using PyPlot
using Interpolations
using LinearAlgebra
using BenchmarkTools
using ProgressBars

usingGpu = false

CuArrays.allowscalar(false)

include("layers.jl")
include("boardGrid.jl")

N_i = 40
N_e = 200
if usingGpu
    layer_stochastic = OneDstpLayer(N_i,N_e,true) |>  gpu
    layer_rate = OneDstpLayer(N_i,N_e,false) |>  gpu
else
    layer_stochastic = OneDstpLayer(N_i,N_e,true)
    layer_rate = OneDstpLayer(N_i,N_e,false)
end

#the location specific input fields :
# field_e = Float32.(rand(Uniform(0,1),N_e))
# field_i = Float32.(rand(Uniform(0,1),N_i))
field_e = Float32.(range(0f0,1f0,length=N_e))
field_i = Float32.(range(0f0,1f0,length=N_i))

speed_field_e = append!([-1f0 for _ in 1:1:div(N_e,2)],[1f0 for _ in 1:1:div(N_e,2)])
speed_field_i = zeros(Float32,N_i)

# generate trajectory over T minutes:
function traj(T,x0 = 0f0)
    #quasi-random trajectory as used in Widlowski-Fiete 2014
    X_sample = zeros(Float32,Int(div(T,dt))+1)
    X_sample[1] = x0
    t = 0f0
    n = 1
    while(n<size(X_sample,1))
        goodStep = false
        v = 0f0
        Δt = 0f0
        while(!goodStep)
            # sample of speed
            v = rand(Uniform(-1f0,1f0),1)[1] #max speed: 1m.s-1
            Δt = rand(Uniform(0f0,0.02f0),1)[1] # sample of speed interval
            if(X_sample[n]+v*Δt<=1f0 &&  X_sample[n]+v*Δt>=0f0)
                goodStep = true
            end
        end
        # integration:
        step_size = n+Int(div(Δt,dt))>size(X_sample,1) ? size(X_sample,1)-n : div(Δt,dt)
        X_sample[n+1:n+Int(step_size)] = X_sample[n] .+ range(1f0,stop=step_size,step=1f0) .* v*dt
        t = t + dt*div(Δt,dt)
        n = n+Int(step_size)
    end
    X_sample = boxCarFilter(X_sample,20)
    X_sample = (X_sample .- minimum(X_sample)) ./ maximum(X_sample)
    V = Float32.(diff(X_sample,dims=1)) ./ dt
    V = append!([0.f0],V)
    inputs_e = map(x->10f0*exp.(-norm.(x .- field_e,2)/(2f0*0.01f0*0.01f0)),X_sample)
     #σ_loc = 1cm = 0.01 m
    inputs_i = map(x->10f0*exp.(-norm.(x .- field_i,2)/(2f0*0.01f0*0.01f0)),X_sample)
    return inputs_e,inputs_i,V,X_sample
end

function traj2(T)
    x_sample = zeros(T+1)
    x_sample[1] = 0
    x_sample[end] = 1
    x_sample[2:end-1] = rand(Uniform(0,1),T-1)
    # add at the begining 0 and 1 to make sure all angles are sampled
    itp = interpolate(x_sample,BSpline(Quadratic(Flat(OnGrid()))))
    t_itp= 0:1:T
    sitp = Interpolations.scale(itp,t_itp)
    tsteps = 0:dt:T
    X = Float32.(sitp.(tsteps))
    V = Float32.(diff(X,dims=1))
    V = append!([0.f0],V)
    inputs_e = map(x->10f0*exp.(-norm.(x .- field_e,2)/(2f0*0.01f0*0.01f0)),X)
     #σ_loc = 1cm = 0.01 m
    inputs_i = map(x->10f0*exp.(-norm.(x .- field_i,2)/(2f0*0.01f0*0.01f0)),X)
    return inputs_e,inputs_i,V,X
end
# let us verify trajectory distribution
T = 60*4
@time inputs_e,inputs_i,V,X = traj(T)
fig,ax = plt.subplots(1,3)
ax[1].hist(V,bins=20)
ax[2].hist(X,bins=20)
X_bin = range(0f0,stop=1f0,length=1000)
speed_binned = map(x -> abs.(V[abs.(X .- x) .< 0.025f0]),X_bin)
ax[3].plot(X_bin,mean.(speed_binned))
display(fig)


β_vel = 0.9f0 # gain of velocity input
a_0 = 60f0 # steepness of the tapering by the envelop
ΔX = 0.72f0 # tapering range...

# training loop
idt = 2000 # 1/dt
# Run for period of 4 minutes (with stdp): 4*60*2000
#   The we "move the animal in a new environment": we reset the neurons synaptic activation
#   at which we record the pop activity for 30 s (no stdp) and compute a grid score over these seconds.
#   Then we reset back the activations to their original value, and put back the animal where he was....

# In the Fiete paper, 1 s is used, but this does not enable to run through the linear track at all,
# because of that the tuning curve would have been very badly evaluated.
board_rate = boardGrid(layer_rate,usingGpu)
board_stoch = boardGrid(layer_stochastic,usingGpu)

env_e = zeros(Float32,N_e)
env_i = zeros(Float32,N_i)
x_finish = [0.0f0]

for periods in 1:1:59 # 60 periods of 4 minutes...
    T=60*4 #T =  4 minutes in seconds
    inputs_e,inputs_i,V,X = traj(T, x_finish[1])
    x_finish[1] = X[end]
    fig,ax = plt.subplots(1,3)
    ax[1].hist(V,bins=20)
    ax[2].hist(X,bins=20)
    X_bin = range(0f0,stop=1f0,length=1000)
    speed_binned = map(x -> abs.(V[abs.(X .- x) .< 0.025f0]),X_bin)
    ax[3].plot(X_bin,mean.(speed_binned))
    display(fig)
    println("training")
    @time for idx in ProgressBar(1:1:T*idt)
        shunt_e = 1f0 .+ β_vel .* V[idx] .* speed_field_e
        shunt_i = 1f0 .+ β_vel .* V[idx] .* speed_field_i
        Xᵉ = abs.(field_e .- 0.5f0)
        env_e[Xᵉ .>= 1f0-ΔX] = exp.(-a_0 .* (Xᵉ[Xᵉ .>= 1f0-ΔX] .- 1f0 .+ ΔX) ./ ΔX)
        env_e[Xᵉ .< 1f0-ΔX] .= 1f0
        Xⁱ = abs.(field_i .- 0.5f0)
        env_i[Xⁱ .>= 1f0-ΔX] = exp.(-a_0 .* (Xⁱ[Xⁱ .>= 1f0-ΔX] .- 1f0 .+ ΔX) ./ ΔX)
        env_i[Xⁱ .< 1f0-ΔX] .= 1f0

        if usingGpu
            layer_stochastic(gpu(inputs_e[idx]),gpu(inputs_i[idx]),gpu(shunt_e),gpu(shunt_i),gpu(env_e),gpu(env_i))
            layer_rate(gpu(inputs_e[idx]),gpu(inputs_i[idx]),gpu(shunt_e),gpu(shunt_i),gpu(env_e),gpu(env_i))
        else
            layer_stochastic(inputs_e[idx],inputs_i[idx],shunt_e,shunt_i,env_e,env_i)
            layer_rate(inputs_e[idx],inputs_i[idx],shunt_e,shunt_i,env_e,env_i)
        end
    end
    prepareBoard(board_rate)
    prepareBoard(board_stoch)
    T=30 #30 seconds
    inputs_e,inputs_i,V,X = traj(T)
    print("testing")
    @time for idx in 1:1:size(X,1)
        shunt_e = 1f0 .+ β_vel .* V[idx] .* speed_field_e
        shunt_i = 1f0 .+ β_vel .* V[idx] .* speed_field_i
        if usingGpu
            board_stoch(gpu(inputs_e[idx]),gpu(inputs_i[idx]),gpu(shunt_e),gpu(shunt_i))
            board_rate(gpu(inputs_e[idx]),gpu(inputs_i[idx]),gpu(shunt_e),gpu(shunt_i))
        else
            board_stoch(inputs_e[idx],inputs_i[idx],shunt_e,shunt_i)
            board_rate(inputs_e[idx],inputs_i[idx],shunt_e,shunt_i)
        end
    end
    computeTC(board_rate,X)
    computeTC(board_stoch,X)
    reset_synapse(layer_rate,board_rate.save_si,board_rate.save_se)
    reset_synapse(layer_stochastic,board_stoch.save_si,board_stoch.save_se)
    #plot boards:
    if periods>1
        myplot(board_rate)
        myplot(board_stoch)
    end
end

plot_tc(board_rate)
plot_tc(board_stoch)

cmap = plt.get_cmap("hot")
fig,ax = plt.subplots()
ax.imshow(abs.(layer_stochastic.W_ii),cmap=cmap)
display(fig)
fig,ax = plt.subplots()
ax.imshow(abs.(layer_stochastic.W_ei),cmap=cmap)
display(fig)
fig,ax = plt.subplots()
ax.imshow(abs.(layer_rate.W_ii),cmap=cmap)
display(fig)

annealScore = W -> sum(map(i->sum(map(j->W[i,j]^2*(i-j)^2,1:1:size(W,2))),1:1:size(W,1)))
function annealing(W,T,recurrent::Bool)
    # execute simulated annealing at temperature T for matrix W
    # If W is not recurrent, we pick randomly x or y change, otherwise permutation should apply to both axis
    display(T)
    Eold = annealScore(W)
    n_accept = 0
    for t in 1:1:10^5
        pos = recurrent ? 1 : rand(1:1:2)
        i,j = rand(1:1:size(W,pos),2)
        perm = collect(range(1,stop=size(W,pos),step=1))
        perm[i] = j
        perm[j] = i
        if pos == 1
            W2  = reduce(hcat,map(s->permute!(s,perm),eachcol(W)))
        else
            W2  = transpose(reduce(hcat,map(s->permute!(s,perm),eachrow(W))))
        end
        E = annealScore(W2)
        if E <= Eold
            W = W2
            Eold = E
            n_accept += 1
        else
            s = rand(Uniform(0,1))
            if s < exp(-(E-Eold)/T)
                Eold = E
                W = W2
                n_accept += 1
            end
        end
        if n_accept==10^2
            return annealing(W,T*0.9,recurrent)
        end
    end
    return W
end

# @time Wii = annealing(layer_stochastic.W_ii,10^8,true)
# fig,ax = plt.subplots()
# ax.imshow(Wii)
# display(fig)
#
# b = board_rate
# b.mem_activ_e
# X_bin = 0:0.05f0:1
# T=30
# ρ_train = transpose(reduce(hcat,b.mem_activ_e))
# inputs_e,inputs_i,V,X = traj(T)
# abs.(X[1:end-1] .- X_bin[1]) .< 0.025f0
# ρ_binned = map(x -> ρ_train[abs.(X[1:end-1] .- x) .< 0.025f0,:],X_bin)
# firing_rates = map(rb->sum(rb,dims=1)/(size(rb,1)),ρ_binned)
# reduce(vcat,firing_rates)
# T=30
# inputs_e,inputs_i,V,X = traj(T)
# idx = 1
# shunt_e = 1f0 .+ β_vel .* V[idx] .* speed_field_e
# shunt_i = 1f0 .+ β_vel .* V[idx] .* speed_field_i
# board_rate(gpu(inputs_e[idx]),gpu(inputs_i[idx]),gpu(shunt_e),gpu(shunt_i))
# computeTC(board_rate,X[idx])
# ρ_train = transpose(reduce(hcat,board_rate.mem_activ_e))
# X_bin = 0:0.05f0:1
# X_input = X
# abs.(X .- X_bin[1]) .< 0.025f0
# ρ_binned = map(x -> ρ_train[abs.(X .- x) .< 0.025f0,:],X_bin)
