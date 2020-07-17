# 🔥 A reproduction of the BVC-SR model of Cothi and Barry
using PyPlot
using LinearAlgebra
using BenchmarkTools
using ProgressBars
using Distributions
using ImageFiltering
using Roots

function angleDiff(e)
    if abs(e+2.0f0*π) < abs(e-2.0f0*π)
        return abs(e+2.0f0*π) < abs(e) ? abs(e+2.0f0*π)  : abs(e)
    else
        return abs(e-2.0f0*π) < abs(e) ? abs(e-2.0f0*π)  : abs(e)
    end
    return abs(e)
end

#boundary fields:
const β = 12f0 # UNIT NOT PROVIDED...
const ξ = 8f0 # UNIT NOT PROVIDED...
const σ_ang = 11.25f0*π/180f0
σ_rad(di) = di/β+ξ
g(di,ϕi) = (r,θ) -> exp(-(r-di)^2/(2f0*σ_rad(di)^2))/(√(2f0*π*σ_rad(di)^2))*exp(-angleDiff(θ-ϕi)^2/(2f0*σ_ang^2))/(sqrt(2f0*π*σ_ang^2))
#note: there might be a unit problem in g for the rad -> deg

DI = Float32.([3.3,10.2,17.5,25.3,33.7,42.6,52.2,62.4,73.3,85.0])
ϕI = Float32.([0,22.5,45,67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5])*π/180f0

const envDim = (60,60)
const γ = 0.995f0
const α_M = 0.0001f0

TC = reduce(vcat,map(di->map(ϕi->g(di,ϕi),ϕI),DI))

dθ = 0.05f0
θ_bin = -π:dθ:π
dx = 1 # bins of 1 cm
OrientedMaps = Float32.(zeros(div(envDim[1],dx),div(envDim[2],dx),size(θ_bin,1)))
function boundary(x,y)
    x==0 ? 1f0 : y ==0 ? 1f0 : x == envDim[1] ? 1f0 : y==envDim[1] ? 1f0 : 0f0
end

#obtain a gaussian filtered boundary map:
Maps = reduce(hcat,map(x->map(y->boundary(x,y),0f0:dx:envDim[1]),0f0:dx:envDim[1]))
# GausMaps = imfilter(Maps, Kernel.gaussian(5))
GausMaps = Maps
# fig,ax = plt.subplots(1,2)
# ax[1].imshow(Maps)
# ax[2].imshow(GausMaps)
# display(fig)

#Next we obtain for each position, the distance to the closest boundary:
minGM = minimum(GausMaps)
maxGM = maximum(GausMaps)
inside(x,y) = x<1 ? false : x>size(GausMaps,1) ? false : y<1 ? false : y>size(GausMaps,2) ? false : true
function bf(θ,x,y)
    function bfr(r)
        posX = round(Int,r*cos(θ)+x)
        posY = round(Int,r*sin(θ)+y)
        return inside(posX,posY) ? GausMaps[posX,posY] : minGM
    end
end
function findDistance(θ,x,y)
    if boundary(x,y)==1f0 #wer remove points that are on the boundary
        return 0f0
    end
    rs = 0f0:dx:2*max(envDim...)
    val = bf(θ,x+1,y+1).(rs)
    #for the moment we only consider the case of one boundary
    # near a boundary we could still have a larger value than further away
    # So to test this we need to check that the derivaive is positive at the putative maximal points
    dval = sign.(diff(val))
    minr = rs[argmax(val[2:end].*dval)+1]
end

dθ = 0.5f0*π/180
θ_bin = -π:dθ:π
X = 0f0:dx:envDim[1]
Y = 0f0:dx:envDim[1]
@time DistancesMaps = map(θ->reduce(hcat,map(x->map(y->findDistance(θ,x,y),X),Y)),θ_bin)

fig = plt.figure(figsize=(10,10))
for i in 1:1:size(θ_bin,1)
    ax = fig.add_subplot(div(size(θ_bin,1),floor(Int,√(size(θ_bin,1)))),28,i)
    ax.imshow(BVCfields[i])
    ax.set_axis_off()
end
display(fig)

#let us obtain the tunign curves:
@time TCs = map(tc->map(θidx->reduce(hcat,map(x->map(y->tc(DistancesMaps[θidx][x,y],θ_bin[θidx]),1:1:size(DistancesMaps[θidx],1)),1:1:size(DistancesMaps[θidx],1))),1:1:size(θ_bin,1)),TC)

TCSumed = map(tc->sum(tc),TCs)
fig = plt.figure(figsize=(10,10))
for i in 1:1:size(TCSumed,1)
    ax = fig.add_subplot(10,16,i)
    ax.imshow(TCSumed[i])
    ax.set_axis_off()
end
display(fig)

const b = 13.02f0 #cm/sec
const μ = -0.03f0*π/180f0 #rad/sec
const σ = 330.12f0*π/180f0 #rad/sec

dt = 1f0/50f0
T = 1:dt:60*60*2 #2 hours, 50 Hz sampling
samples = size(T,1)
RandomTurn = Float32.(rand(Normal(μ,σ),samples))
RandomVel = Float32.(rand(Rayleigh(σ),samples))
Position = zeros(samples,2)
Position[1,:] = [2f0,2f0]
Velocity = zeros(samples,2)

#function to obtain the minimal distance to any wall:
function minDist(x,y)
    dists = map(dm->dm[round(Int,x+1),round(Int,y+1)],DistancesMaps) # Would not work in a more complex environment....
    return minimum(dists),θ_bin[argmin(dists)]
end
currentAngle = [0f0]
v = [20f0]
function step!(Position,currentAngle,v,step)
    print(Position[step-1])
    dWall,aWall = minDist(Position[step-1,:]...)
    println("  ",dWall,"  ",angleDiff(aWall-currentAngle))
    newangle = 0
    if dWall<0.2 && angleDiff(aWall-currentAngle)<π/2f0
        newangle =  sign(angleDiff(aWall-currentAngle))*(π/2f0-angleDiff(aWall-currentAngle))
        v = v-0.5f0*(v-5f0)
    else
        v = RandomVel[step]
        newangle = RandomTurn[step]
    end
    Dir = Vector([cos(currentAngle),sin(currentAngle)])
    Position[step,:] .= Position[step-1] .+ (Dir.*v.*dt)
    currentAngle = mod(currentAngle+newangle+π,2π)-π
    return currentAngle,v
end

for idx in ProgressBar(2:1:samples)
    step!(Position,currentAngle[1],v[1],idx)
end

fig,ax = plt.subplots()
# ax.hist(RandomTurn*dt)
ax.scatter(Position[:,1],Position[:,2])
display(fig)
