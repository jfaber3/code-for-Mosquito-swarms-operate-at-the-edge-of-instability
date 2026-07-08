import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, ConvexHull
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.ndimage import uniform_filter, gaussian_filter
import matplotlib.colors as mcolors
import code
from scipy.optimize import curve_fit
from scipy.stats import weibull_min
from scipy.special import gamma as Gamma
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


#creates function used for updating positions/orientions of all swarm memebers
def make_updater_function(pos, orient, L, gamma, eta):
	N = pos.shape[0]
	def update_insects():
		nonlocal pos, orient
		tree = cKDTree(pos)
		Dists, Nearest_indexes = tree.query(pos, k=N)
		Dists, Nearest_indexes = Dists[:, 1:], Nearest_indexes[:, 1:]   #removing self as neighbor
		WW = 1/(np.maximum(Dists, 1e-12)**gamma)     #weight matix (1/r^gamma)
		WW /= np.sum(WW, axis=1)[:, np.newaxis]   #normalizing
		neighbor_pos = pos[Nearest_indexes]        # shape (N, k-1, 2)
		mf_pos = np.sum(WW[:, :, None] * neighbor_pos, axis=1)
		vx, vy = np.cos(orient), np.sin(orient)
		Dx, Dy = mf_pos[:, 0] - pos[:, 0], mf_pos[:, 1] - pos[:, 1]
		dtheta = np.pi*(Dy*vx - Dx*vy)/L     #scalar regjection of D onto v
		orient += np.clip(dtheta, -np.pi, np.pi) + eta*np.random.uniform(-np.pi, np.pi, size=N) #ensures determinisitc part of turn <= pi
		cos_orient, sin_orient = np.cos(orient), np.sin(orient)
		pos[:, 0] += cos_orient
		pos[:, 1] += sin_orient
		return cos_orient, sin_orient
	return update_insects



#clrs  =["#1100ff", "#00d5ff", "#00ff62", "#ffaa00", "#ff0000", "#ae00ff"]
clrs2  =["#006aff", "#00ff62", "#ffaa00", "#ae00ff"]

time_steps     = 1000    #1k
time_steps_cut = 1000    #additional data for transient 1k

N      =  20
L      =  100

Gammas = np.linspace(0, 3, 31)       #np.linspace(0, 3, 31)
Etas   = np.linspace(0, 0.3, 31)     #np.linspace(0, 0.3, 31)
num_trials = 11  #11


Swarm_Sizes = np.zeros((len(Gammas), len(Etas), num_trials))
Cor_Lens    = np.zeros((len(Gammas), len(Etas), num_trials))
for g in range(len(Gammas)):
	for et in range(len(Etas)):
		gamma, eta = Gammas[g], Etas[et]
		for trial in range(num_trials):
			Rs = np.zeros((time_steps, N))
			Angles = np.zeros((time_steps, N))

			#Scale and Initial Conditions Parameters
			Angs_ic = np.random.uniform(-np.pi, np.pi, size=N)
			Rs_ic = np.random.uniform(0, 1, size=N)**0.5 * (L/2)   #initializing positions on a disk
			pos = np.zeros((N, 2))
			pos[:, 0], pos[:, 1] = Rs_ic*np.cos(Angs_ic), Rs_ic*np.sin(Angs_ic)
			orient = np.random.uniform(-np.pi, np.pi, size=N)
			update_insects = make_updater_function(pos, orient, L, gamma, eta)  #making updater function

			#running transient
			for i in range(time_steps_cut):
				update_insects()

			#collecting data
			for i in range(time_steps):
				update_insects()
				x_cm, y_cm = pos[:, 0].mean(), pos[:, 1].mean()
				Rs[i, :] = np.sqrt( (pos[:, 0] - x_cm)**2 + (pos[:, 1] - y_cm)**2 )
				Angles[i, :] = orient

			#finding correlation length
			cor_steps = 50
			C = np.zeros(cor_steps)
			for tau in range(cor_steps):
				C[tau] = np.mean( np.cos(Angles[tau:, :] - Angles[:time_steps-tau, :]) )
			#plt.figure(1)
			#plt.plot(C, "o-", color=clrs2[g])
			for tau in range(cor_steps):
				if C[tau] < 1/np.e:
					c1, c2, t1, t2 = C[tau-1] - 1/np.e,  C[tau] - 1/np.e,  tau-1,  tau  #now find intersection with x-axis
					cor_len = t1 - c1*( (t2-t1)/(c2-c1) )
					break
			Swarm_Sizes[g, et, trial], Cor_Lens[g, et, trial] = Rs.mean(), cor_len
		print(gamma, eta)



np.savez(r"C:\Users\Justin\Desktop\2D_model_R0_CorLen_data_N=20.npz", Gammas=Gammas, Etas=Etas, Swarm_Sizes=Swarm_Sizes, Cor_Lens=Cor_Lens)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5,5))
plt.subplots_adjust(left=0.05, right=0.97, top=1.0, bottom=0.05)

X, Y = np.meshgrid(Gammas, Etas)
Z = np.median(Swarm_Sizes, axis=2).T

#custom_cmap = mcolors.LinearSegmentedColormap.from_list('bog', ["#f6f6f6", "#00d9ff", "#C800FF"], N=15)  #N=256, 15
surf = ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=False)
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$\eta$")
ax.zaxis.label.set_rotation(180)
ax.set_zlabel("Swarm Radius", labelpad=5)
ax.view_init(elev=30, azim=-135)
ax.set_box_aspect((1, 1, 1.1))






#Loading And Plotting Data
Data = np.load(r"C:\Users\Justin\Desktop\Swarming\Mosquito Swarm Data\2D_model_R0_CorLen_data_N=20.npz")
Gammas0, Etas0, Swarm_Sizes0, Cor_Lens0 = Data["Gammas"], Data["Etas"], Data["Swarm_Sizes"], Data["Cor_Lens"]
X, Y = np.meshgrid(Gammas0, Etas0)
Z1 = np.median(Swarm_Sizes0, axis=2).T
Z2 = np.median(Cor_Lens0, axis=2).T

fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5,5))
plt.subplots_adjust(left=0.05, right=0.97, top=1.0, bottom=0.05)
surf = ax1.plot_surface(X, Y, Z1, cmap='jet', linewidth=0, antialiased=False)
ax1.set_xlabel(r"$\gamma$")
ax1.set_ylabel(r"$\eta$")
ax1.zaxis.label.set_rotation(180)
ax1.set_zlabel("Swarm Radius", labelpad=5)
ax1.view_init(elev=30, azim=-135)
ax1.set_box_aspect((1, 1, 1.1))
plt.savefig("C:/Users/Justin/Desktop/Swarm_Sizes.pdf", dpi=300)

fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5,5))
plt.subplots_adjust(left=0.0, right=0.92, top=1.0, bottom=0.05)
surf = ax2.plot_surface(X, Y, Z2, cmap='jet', linewidth=0, antialiased=False)
ax2.set_xlabel(r"$\gamma$")
ax2.set_ylabel(r"$\eta$")
ax2.zaxis.label.set_rotation(180)
ax2.set_zlabel("Correlation Length", labelpad=5)
ax2.view_init(elev=30, azim=135)
ax2.set_box_aspect((1, 1, 1.1))
plt.savefig("C:/Users/Justin/Desktop/Cor_Lens.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after











