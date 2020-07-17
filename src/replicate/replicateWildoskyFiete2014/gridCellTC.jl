using PyPlot


function tc(θ1,θ2)
    ρ(u1,u2) = exp(cos(u1-θ1)+cos(u2-θ2))
end

r = tc(0.0,0.0)

rCart(x,y) = r(x,y*sin(π/3)) 
