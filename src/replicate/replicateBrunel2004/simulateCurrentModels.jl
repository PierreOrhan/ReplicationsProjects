# Replication of the figures from Brunel's paper.
using Distributions
using Flux
using LinearAlgebra
using ProgressBars
using PyPlot
using CSV,DataFrames
using Optim
include("currentModels.jl")

path = joinpath("src","replicate","replicateBrunel2004")

# FIGURE 7,
# to which we add an additional plot to see the onset of the bump in the rate map.
# We note that the author reported a Tuning Curve w.r.t an angle, but what we think that they really meant
# given the position of the figure reference in the text was the tuning curve with respect to the neurons,
#and thus the x-axis should be the neurons id.
α = 50f0*π/180f0
L0 = 0.02f0 #0.02f0
n = Network(1000,1000,α,L0)
T = 15
raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r = simulate(n,T)

#plot tuning curves
θ_bin_e = range(-π,stop=π,length=n.N_e)
tc_e = sum(raster_e,dims=2)/T
θ_bin_r = range(-π,stop=π,length=n.N_r)
tc_r = sum(raster_r,dims=2)/T
θ_bin_l = range(-π,stop=π,length=n.N_l)
tc_l = sum(raster_l,dims=2)/T
fig,ax = plt.subplots(2,1)
ax[1].plot(θ_bin_e,tc_e,c=(0,0,0))
ax[1].set_ylabel("Mean firing rate (spikes/s)")
ax[1].set_xlabel("neurons prefered angle, rad")
ax[2].plot(θ_bin_r,tc_r,c="b",label="right DTN")
ax[2].set_ylabel("Mean firing rate (spikes/s)")
ax[2].set_xlabel("neurons prefered angle, rad")
ax[2].plot(θ_bin_l,tc_l,c="r",label="left DTN")
ax[2].set_ylabel("Mean firing rate (spikes/s)")
ax[2].set_xlabel("neurons prefered angle, rad")
ax[2].legend()
fig.tight_layout()
display(fig)
fig.savefig(joinpath(path,"figures","figures7a.png"))
#plot rasters
time = range(0,stop=T,length=size(raster_e,2))
events_e = map(r->time[r.==1],eachrow(raster_e))
events_r = map(r->time[r.==1],eachrow(raster_r))
events_l = map(r->time[r.==1],eachrow(raster_l))
events_inputs_r = map(r->time[r.==1],eachrow(spikePerInterval_r))
events_inputs_l = map(r->time[r.==1],eachrow(spikePerInterval_l))
events_inputs_e = map(r->time[r.==1],eachrow(spikePerInterval_e))
fig,ax = plt.subplots(3,1,figsize=(10,10))
ax[1].eventplot(events_e)
ax[1].set_ylabel("LMN neurons id")
ax[2].eventplot(events_r)
ax[2].set_ylabel("DTN neurons id, right")
ax[3].eventplot(events_l)
ax[3].set_ylabel("DTN neurons id, left")
ax[3].set_xlabel("time (s)")
fig.tight_layout()
display(fig)
fig.savefig(joinpath(path,"figures","figures7b"))


##Figure 8
function decoder(raster_e,θ_bin_e)
    # Decoder using the population vector (here defined as the vector of size the number of neurons and value  0 or 1, with 1 for a spiking neurons)
    #raster_e: the rasters at each time step
    #θ_bin_e : the prefere angle (rad)
    # here arctan would not directly work. we need to use atan2 (see wikipedia) to obtain the right sign!
    θ_decoded = atan.(sum(sin.(θ_bin_e).*raster_e,dims=1),sum(cos.(θ_bin_e).*raster_e,dims=1)) #output angle in [-π,π]
end

L0s = 0.03f0:0.002f0:0.15f0
bump_angle_over_time = []
for tpl in enumerate(L0s)
    idx,L0 = tpl
    # each run take about 40 s so the total runing time for this loop is about 40 minutes.
    α = 50f0*π/180f0
    n = Network(1000,1000,α,L0)
    T = 5
    raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r = simulate(n,T)
    d = decoder(raster_e,θ_bin_e)[1,:]
    push!(bump_angle_over_time,d)
    if L0 in [0.03f0,0.09f0,0.062f0]
        time = range(0,stop=T,length=size(raster_e,2))
        fig,ax = plt.subplots()
        ax.scatter(time,mod.(d,2f0*π).*180f0/π,s=0.2,marker="x")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Encoded head direction (deg)")
        display(fig)
        fig.savefig(joinpath(path,"figures",string("8b",Int(idx),".png")))
    end
end
CSV.write(joinpath(path,"figures","data.csv"),DataFrame(bump_angle_over_time))

bump_angle_over_time = CSV.read(joinpath(path,"figures","data.csv")) |> DataFrame
bump_angle_over_time = transpose(convert(Matrix,bump_angle_over_time))

function angleToR(angles)
    θ = [angles[1]]
    for tpl in enumerate(angles[2:end])
        idx,a = tpl
        if a>angles[idx]+π
            push!(θ,θ[end]-angleDiff(a-angles[idx]))
        elseif a<angles[idx]-π
            push!(θ,θ[end]+angleDiff(a-angles[idx]))
        else
            push!(θ,θ[end]+sign(a-angles[idx])*angleDiff(a-angles[idx]))
        end
    end
    return θ
end
function angle_fit(bump_angle_over_time,idxt0,idxtend)
    real_line_angles = angleToR.(collect(eachrow(bump_angle_over_time[:,idxt0:idxtend])))
    vel = []
    for r in real_line_angles
        X2 = zeros(size(r,1),2)
        X2[:,1] = times[idxt0:idxtend]
        X2[:,2] = ones(size(X2,1))
        α = (X2'*X2) \ (X2' * r)
        push!(vel,α[1])
    end
    return vel
end

Ts = range(1000,stop=size(bump_angle_over_time,2),step=150)
vels = []
for tpl in enumerate(Ts[2:end])
    push!(vels,angle_fit(bump_angle_over_time,Ts[tpl[1]],tpl[2]))
end
vels= abs.(reduce(hcat,vels))
mean_vels = mean(vels,dims=2)
var_vels = var(2*180/T*vels,dims=2)

fig,ax = plt.subplots()
ax.plot(L0s,2*180*mean_vels/T)
ax.set_ylabel("Mean bump velocity(deg/s)")
ax2 = ax.twinx()
ax2.plot(L0s,var_vels,linestyle="--")
ax2.set_ylabel("Velocity veriance (deg/s)^2")
ax.set_xlabel("connection weight between DTN networks")
display(fig)
fig.savefig(joinpath(path,"figures","8a.png"))

# Figure 9
Hd_vel = 0f0:20f0:1000f0 #deg/s
L0 = 0.02f0
nb_trial = 5
mean_bump_velocity = []
for v in Hd_vel
    mb = []
    T = 5
    for id_trail  in 1:1:nb_trial
        α = 50f0*π/180f0
        n = Network(1000,1000,α,L0)
        # 🔥 v is in degree here... to be accounted  for in the conversion to Ampers in the network.
        raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r = simulate(n,T,v*π/180f0)
        θ_bin_e = range(-π,stop=π,length=n.N_e)
        d = decoder(raster_e,θ_bin_e)
        push!(mb,d)
    end
    decoded_angles = reduce(vcat,mb)
    push!(mean_bump_velocity,mean(abs.(angle_fit(decoded_angles,1000,size(decoded_angles,2))))*2*180/(T-dt*1000))
end
CSV.write(joinpath(path,"figures","dataFig9a.csv"),DataFrame([mean_bump_velocity]))
mean_bump_velocity = CSV.read(joinpath(path,"figures","dataFig9a.csv")) |> DataFrame
mean_bump_velocity = transpose(convert(Matrix,mean_bump_velocity))[1,:]

nb_trial=5
mean_bump_velocity2 = []
L0 = 0.02f0
for v in Hd_vel
    mb = []
    T = 5
    for id_trail  in 1:1:nb_trial
        α = 65*π/180f0
        n = Network(1000,1000,α,L0)
        # 🔥 v is in degree here... to be accounted  for in the conversion to Ampers in the network.
        raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r = simulate(n,T,v*π/180f0)
        θ_bin_e = range(-π,stop=π,length=n.N_e)
        d = decoder(raster_e,θ_bin_e)
        push!(mb,d)
    end
    decoded_angles = reduce(vcat,mb)
    push!(mean_bump_velocity2,mean(abs.(angle_fit(decoded_angles,1000,size(decoded_angles,2))))*2*180/(T-dt*1000))
end
CSV.write(joinpath(path,"figures","dataFig9b.csv"),DataFrame([mean_bump_velocity2]))
mean_bump_velocity2 = CSV.read(joinpath(path,"figures","dataFig9b.csv")) |> DataFrame
mean_bump_velocity2 = transpose(convert(Matrix,mean_bump_velocity2))[1,:]

fig,ax= plt.subplots()
ax.plot(Hd_vel,Hd_vel,linestyle="--",c=(0,0,0))
ax.scatter(Hd_vel,mean_bump_velocity,marker="o",c=(0,0,0),label="α=50")
ax.scatter(Hd_vel,mean_bump_velocity2,marker="x",c=(0,0,0),label="α=65")
ax.set_xlabel("Head angular velocity (deg/s)")
ax.set_ylabel("Bump velocity (deg/s)")
# ax.set_ylim(1000)
fig.legend()
display(fig)
fig.savefig(joinpath(path,"figures","9.png"))

⃨
# Figure 10
# The data for the rat head direction was not available, so I used one from mice in the lab
miceTracking  = CSV.read(joinpath(path,"data","Tracking_data.csv")) |> DataFrame
# in the 4 column is the head direction, in the first we can see the time steps, in seconds
miceHd = miceTracking[:,4]
miceTsteps = miceTracking[:,1]
InvdtRecording = 100 #recording time steps were approximately 0.01 s
speedData = sign.(diff(miceHd[1:120*InvdtRecording]).*π/180f0).*angleDiff.(diff(miceHd[1:120*InvdtRecording].*π/180f0))./diff(miceTsteps[1:120*InvdtRecording])
angles = miceHd[1:120*InvdtRecording]
##Safety check
fig,ax = plt.subplots()
ax.plot(speedData)
display(fig)
# We observe thaat two speed values are > 100 rad/s, we correct for them:
speedData[abs.(speedData).>100] .= 0f0
fig,ax = plt.subplots()
ax.plot(speedData)
display(fig)
# We observe
#For the simulation we will simulate with time-steps of dt s, dt = 0.5ms. SO we need to interpolate the speedData array
using Interpolations
T = 120
α = 50*π/180f0
L0 = 0.02f0
n = Network(1000,1000,α,L0)
speedFilled = zeros(Float32,Int(div(T,dt)))
#🔥 div(T,dt) as used previously is badly rounded, same for dtRecording/dt ! So we use the value: 20
# we forward fill the speed point between each step:
itp = interpolate(speedData, BSpline(Constant()))
speedFilled .= itp.(range(1,stop=T*InvdtRecording-1,length=size(speedFilled,1)))

raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r = simulate(n,T,speedFilled)
#the simulation took 25 minutes 39 sc
θ_bin_e = range(-π,stop=π,length=n.N_e)
d = decoder(raster_e,θ_bin_e)
#We then anchor the decoded signal to the initial signal to do so we
# can either take the angle different that minimizes the mean signed angular distance
# or make sure the error is 0 initially.

#Minimizing the mean error:
bumpDiff = d[1:20:end].-angles[1:1:end].*π/180
angleDiffWithGood = mean(sign.(bumpDiff).*angleDiff.(bumpDiff))
d .= mod.(d .+ angleDiffWithGood .+π,2π).-π
fig,ax = plt.subplots()
ax.scatter(1:1:size(d,2),d[1,:],s=0.2f0,label="prediction (oversampled)")
ax.scatter(1:20:size(d,2),angles.*π/180,s=0.2f0,label="true angle (undersampled)")
ax.set_xlabel("time steps, true angle present every 20 time step")
ax.set_ylabel("angle, rad")
display(fig)
fig.savefig(joinpath(path,"figures","10_minimizeMeanError.png"))
# we downsample the computed angle:
angle_pred = d[1:20:size(d,2)]
fig,ax = plt.subplots()
ax.plot(0:0.01:T-0.01,angleDiff.(angle_pred.-angles.*π/180))
ax.set_xlabel("time (s)")
ax.set_ylabel("absolute error (rad)")
display(fig)
fig.savefig(joinpath(path,"figures","11_minimizeMeanError.png"))

θ_bin_e = range(-π,stop=π,length=n.N_e)
d = decoder(raster_e,θ_bin_e)
#Making sure no error initially:
bumpDiff = d[10].-angles[10].*π/180
angleDiffWithGood = sign.(bumpDiff).*angleDiff.(bumpDiff)
d .= mod.(d .+ angleDiffWithGood .+π,2π).-π
fig,ax = plt.subplots()
ax.scatter(1:1:size(d,2),d[1,:],s=0.2f0,label="prediction (oversampled)")
ax.scatter(1:20:size(d,2),angles.*π/180,s=0.2f0,label="true angle (undersampled)")
ax.set_xlabel("time steps, true angle present every 20 time step")
ax.set_ylabel("angle, rad")
display(fig)
fig.savefig(joinpath(path,"figures","10_minimizeInitialError.png"))
# we downsample the computed angle:
angle_pred = d[1:20:size(d,2)]
fig,ax = plt.subplots()
ax.plot(0:0.01:T-0.01,angleDiff.(angle_pred.-angles.*π/180))
ax.set_xlabel("time (s)")
ax.set_ylabel("mean tracking error (rad) (n=1)")
display(fig)
fig.savefig(joinpath(path,"figures","11_minimizeInitialError.png"))

# next  work on fig 12

T = 1.5
α = 50*π/180f0
L0 = 0.02f0
n = Network(1000,1000,α,L0)
raster_e,raster_r,raster_l,spikePerInterval_e,spikePerInterval_l,spikePerInterval_r = simulate_external_inputs(n,T,π)

#computation of the instantaneous firing rate in bins of 10 ms:
firing_rates = map(idx->sum(Float32.(raster_e[:,idx*20+1:min((idx+1)*20,size(raster_e,2))]),dims=2)[:,1]./(20f0*dt),1:1:Int(div(T,dt*20)))
group_1 = 400:600
group_2 = 600:800   #parameters to change manually for each plot (very easy)
mean_fr_group1 = map(fr->mean(fr[group_1]),firing_rates)
mean_fr_group2 = map(fr->mean(fr[group_2]),firing_rates)
#normalize the spike count:
min0 = mean(mean_fr_group2[50:90])
max0 = mean(mean_fr_group2[110:149])
normalized_group_1 = (mean_fr_group1.-min0)./(max0-min0)
normalized_group_2 = (mean_fr_group2.-min0)./(max0-min0)

times = range(0,stop=T,length=size(raster_e,2))
events_e = map(r->times[r.==1],eachrow(raster_e))
fig,ax = plt.subplots(2,1,figsize=(10,10))
ax[1].eventplot(events_e)
ax[1].set_ylabel("LMN neurons id")
ax[1].set_xlabel("time (s)")
ax[2].plot(time[1:20:size(times,1)-20],normalized_group_1)
ax[2].plot(time[1:20:size(times,1)-20],normalized_group_2)
ax[2],set_ylabel("Normed spike count")
ax[2].set_xlable("time(s)")
fig.tight_layout()
display(fig)
fig.savefig(joinpath(path,"figures","figures12.png"))

#Note: for fig 12 the bump was not necessarily reorienting at the asked position (180°)
