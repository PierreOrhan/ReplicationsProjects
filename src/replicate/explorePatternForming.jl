using PyPlot
using Flux
using LinearAlgebra
using BenchmarkTools
using ProgressBars
using Distributions
using Roots
using ForwardDiff
using DifferentialEquations

angleDiff(e) = min(abs(e),2f0*π-abs(e))
θ_bin_e = Float32.(range(-π,stop=π,length=100))
θ_bin_i = Float32.(range(-π,stop=π,length=100))

function get_σe(σ_e)
    function get_ω(ω)
        L0 = 0.01f0
        θ_0 = 0
        Ii  = 40f0
        θ_bin_e = Float32.(range(-π,stop=π,length=1000))
        θ_bin_i = Float32.(range(-π,stop=π,length=100))
        IE = 10f0
        WI = 100f0/(size(θ_bin_e,1))
        r = map(θ->Ii+WI*sum((cos.(ω.*(θ_bin_e.-θ))).*exp.(- (cos.(0.5f0 .*(θ_bin_e.-θ_0))).^2/(2f0*σ_e^2))),θ_bin_i)
    end
    rs = get_ω.(1f0:1f0:5f0)
    e = reduce(hcat,rs)
    return e[1,:]
end
l = 10f0:5f0:40f0
spTc = get_σe.(l.*π/180f0)
spTc = transpose(reduce(hcat,spTc))

using MultivariateStats
θ_bin = Float32.(range(-π,stop=π,length=150))
N_e = 1000
Xis = zeros(Float32,(size(θ_bin,1),N_e))
θ_bin_e = Float32.(range(0,stop=2*π,length=N_e))
angleDiff(e) = min(abs(e),2f0*π-abs(e))
σ_e = 60f0*π/180f0
using FFTW
function spectr(σ_e)
    g(x,θ) = exp(-cos(1/2*(θ_bin_e[x]-θ))^2/(2f0*σ_e^2))
    # g(x,θ) = exp(-angleDiff(θ_bin_e[x]-θ)^2/(2f0*σ_e^2))
    X= map(θ->map(x->g(x,θ),1:1:size(Xis,2)),θ_bin)
    Xis .= transpose(reduce(hcat,X))
    fft_res = map(xis->map(n->sum(xis.*exp.(-n*im.*θ_bin_e)),1:1:div(size(θ_bin_e,1),2)),eachrow(Xis))
    Norm_spectrum = reduce(hcat,map(n->real.(n),fft_res))
    return Norm_spectrum,transpose(reduce(hcat,X))
end

l = 10f0:5f0:40f0
res = spectr.(l.*π/180f0)
fft_res = map(r->r[2],res)
res = map(r->r[1],res)
Normed_spectrums = reduce(hcat,res)
fig,ax = plt.subplots(3,1,figsize=(10,10))
for tpl in enumerate(res)
    idx,r = tpl
    ax[1].plot(fft_res[idx][1,:],c=cmap(idx/size(l,1)))
    ax[1].set_xlabel("neurons idx")
    ax[1].set_ylabel("firing rate (a.u), bump in first position")
    ax[2].plot(r[1:1:5,1]./sum(r[1:1:5,1]),c=cmap(idx/size(l,1)),label=string("σ_e =",l[idx],"°"))
    ax[2].set_xlabel("frequency")
    ax[2].set_ylabel("Power (norm of the 5th first FFT coefficient) \n normalized")
    ax[2].set_xticks(0:1:4)
    ax[2].set_xticklabels(1:1:5)
    ax[3].plot(0:1:size(spTc,2)-1,(spTc[idx,:].-Ii)./sum(spTc[idx,:].-Ii),linestyle="--",c=cmap(idx/size(l,1)))
    ax[3].set_ylabel("corrected firing rate of the first FS cell \n in each subpopulation, normalized")
    ax[3].set_xlabel("subpopulation index, coding for a particular frequency")
    ax[3].set_xticks(0:1:4)
    ax[3].set_xticklabels(1:1:5)
end
fig.legend()
fig,tight_layout()
display(fig)
fig.savefig("spectrum1.png")
