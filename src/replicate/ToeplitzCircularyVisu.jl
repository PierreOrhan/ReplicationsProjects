using LinearAlgebra
using PyPlot
using Distributions
using Random


λ(j,C) = 1/sqrt(size(C,2))*sum(map(k->C[1,k+1]*(cos(2*π*k*j/size(C,2))+sin(2*π*k*j/size(C,2))*im),0:1:size(C,2)-1))

binAngle = Vector(rand(Uniform(0,2π),1000))
binAngle = Vector(1:0.01:2π)
shuffle!(binAngle)
HDField = 0:0.1:2*π
P2 =reduce(hcat,map(ϕ->map(x->exp(cos(x-ϕ)),binAngle),HDField))
fig,ax = plt.subplots()
ax.imshow(P)
display(fig)
fig,ax = plt.subplots()
ax.imshow(P2*transpose(P2))
display(fig)
fig,ax = plt.subplots()
P = reduce(hcat,map(ϕ->map(x->exp(cos(x-ϕ)),sort(binAngle)),HDField))
ax.imshow(P*transpose(P))
display(fig)
# Λ = map(j->λ(j,C),0:1:size(C,2)-1)
# J = norm.(Λ)
# Λ2 = map(j->λ(j,C2),0:1:size(C2,2)-1)
# J2 = norm.(Λ2)
C = P*transpose(P)
eigC = eigen(C)
C2 = P2*transpose(P2)
eigC2 = eigen(C2)
fig,ax = plt.subplots()
ax.plot(eigC.values[1:500])
ax.plot(eigC2.values[1:500])
display(fig)

posX = -2.2:0.1:2.2
posY = -2.2:0.2:2.2
generatePos(x) = rand(Uniform(-2.2,2.2),2)
# binPos = sort(Vector(generatePos.(1:1:1000)))
binPos = map(x->map(y->Vector([posX[x],posY[y]]),1:1:size(posY,1)),1:1:1size(posX,1))
binPos = reduce(vcat,binPos)
PosField = sort(Vector(generatePos.(1:1:2000)))
#Tuning curves for positions
gausstc(x,c) = exp(-sum((x-c).^2)/(2*0.5)) - exp(-sum((x-c).^2)/(2*4))
GaussTC = reduce(hcat,(map(c->map(x->gausstc(x,c),binPos),PosField)))
fig,ax = plt.subplots()
ax.imshow(GaussTC)
display(fig)
#Plot first line of correlation matrix:
V = reshape(GaussTC[1,:],(1,size(GaussTC,2)))
C = (V*transpose(GaussTC))
fig,ax = plt.subplots()
ax.plot(C[1,:])
display(fig)
C2 = (GaussTC*transpose(GaussTC))
fig,ax = plt.subplots()
ax.imshow(C2)
display(fig)

eigC = eigen(C2)
eigCvals= eigC.values
eigCvects = eigC.vectors
eigC2dVects = map(e->sum(map(i->Vector(e[i]),1:1:size(binPos,1))),eachrow(eigCvects))
eigC2dX = map(e->e[1],eigC2dVects)
eigC2dY = map(e->e[2],eigC2dVects)
using3D()
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(eigC2dX[1:end-1],eigC2dY[1:end-1],eigC.values[1:end-1])
display(fig)

fig,ax = plt.subplots()
ax.plot(eigC.values)
display(fig)

eigOfeigC =  eigen(eigCvects)

using SpecialMatrices
M = abs.(Toeplitz(collect(-50:50)))
eigM =eigen(M)
fig,ax = plt.subplots()
ax.imshow(M)
display(fig)
