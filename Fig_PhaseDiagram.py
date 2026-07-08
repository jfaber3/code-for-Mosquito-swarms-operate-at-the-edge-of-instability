import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import code
from scipy.special import gamma as GAMMA



Data1 = np.load(r"C:\Users\Justin\Desktop\Swarming\Manuscript_Figs\Heatmap_Data_N=20_trials=31.npz")
Data2 = np.load(r"C:\Users\Justin\Desktop\Swarming\Manuscript_Figs\Heatmap_Data_N=40_trials=25.npz")
Data3 = np.load(r"C:\Users\Justin\Desktop\Swarming\Manuscript_Figs\Heatmap_Data_N=80_trials=15.npz")
Data4 = np.load(r"C:\Users\Justin\Desktop\Swarming\Manuscript_Figs\Heatmap_Data_N=160_trials=5.npz")



custom_cmap = mcolors.LinearSegmentedColormap.from_list('bog', ["#00AAFF", "#FF0000FF", "#F6FF00FF"], N=256)  #N=256, 15
fig = plt.figure(figsize=(5, 4), constrained_layout=True)
ax1  = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax2  = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax3  = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax4  = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)




def Plot(Data, ax, N):
    Gammas, Etas, Powers = Data["Gammas"], Data["Etas"], Data["Powers"]
    Powers = gaussian_filter(Powers, sigma=1, mode='nearest')
    im = ax.imshow(Powers.T, cmap=custom_cmap, origin='lower', interpolation=None, extent=[ Gammas[0], Gammas[-1], Etas[0], Etas[-1] ], aspect='auto', vmin=0.0, vmax=2.0) #cmaps: brg, tab10
    G, E = np.meshgrid(Gammas, Etas)
    ax.set_title("$N =$" + str(N))
    if ax == ax4:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\alpha$")

    #BIFURCATION CURVES
    x_thry = np.linspace(1.000000001, 6, 10000)
    l = 1
    L = 100
    Rs = np.sqrt(l*L/np.pi)
    a = Rs * np.sqrt( 2*np.log(2) )    #separation threshold
    r0 = 2*Rs   #9.377 + 1.239*x_thry**2.462   #empirical relationship found below     (or use 2Rs)

    #MAKING UPPER BIFURCATION CURVE
    d = a   #use d=a or d=2Rs or d=a/2
    arg_simple = (r0/d)*N**(1/(1-x_thry))
    eta_c_simple = 1 - (1/np.pi)*np.arcsin(arg_simple)
    ax.plot(x_thry, eta_c_simple, "-", color="#A3A3A3", linewidth=2)
    ax.set_ylim(0, 1)

    #MAKING LOWER BIFURCATION CURVE
    Brackets = GAMMA(x_thry)*(r0/a)**(2*x_thry-3) - (3/(4*N**2))*GAMMA(2*x_thry-1)*(r0/a)**(4*x_thry-5) + (5/(8*N**4))*GAMMA(3*x_thry-2)*(r0/a)**(6*x_thry-7)
    y_thry_lower = np.sqrt(  (l/(32*np.pi*r0)) * Brackets )
    y_thry_lower_infinity = np.sqrt(  (l/(32*np.pi*r0)) * GAMMA(x_thry)*(r0/a)**(2*x_thry-3) )    #Theoretical N=infinity curve
    #if ax == ax4:
    ax.plot(x_thry, y_thry_lower_infinity, ":", color="#000000", linewidth=2)
    ax.plot(x_thry, y_thry_lower, color="#000000", linewidth=2)




Plot(Data1, ax1, 20)
Plot(Data2, ax2, 40)
Plot(Data3, ax3, 80)
Plot(Data4, ax4, 160)


#Experimental Data points
Gamma_exp = np.array([2.33,    2.4,  2.35,  2.5,   2.5,  2.4])
Eta_exp   = np.array([0.093, 0.085, 0.085, 0.07, 0.095, 0.08])
ax1.plot(Gamma_exp, Eta_exp, "*", color ="#dd00ff",  ms=5)


ax3.set_xlabel(r"Mean field weight exponent ($\gamma$)")
ax3.set_ylabel(r"Noise strength ($\eta$)")


plt.savefig("C:/Users/Justin/Desktop/Fig-Phase_Diagram.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after



'''
#finding breaking gamma for female chase plots, N=infinity limit
x_thry = np.linspace(1.000000001, 6, 10000)
l = 1
L = 100
Rs = np.sqrt(l*L/np.pi)
a = Rs * np.sqrt( 2*np.log(2) )    #separation threshold
r0 = 2*Rs   #9.377 + 1.239*x_thry**2.462   #empirical relationship found below     (or use 2Rs)
y_thry = np.sqrt(  (l/(32*np.pi*r0)) * GAMMA(x_thry)*(r0/a)**(2*x_thry-3) )
plt.figure()
plt.plot(x_thry, y_thry, "o-")
plt.axhline(y=0.02)
plt.axhline(y=0.04)
plt.axhline(y=0.085)
plt.show()
'''





