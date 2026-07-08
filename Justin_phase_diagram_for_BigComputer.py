import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import code



#creates function used for updating positions/orientions of all swarm members
def make_updater_function(pos, orient, L, gamma, eta):
	N = pos.shape[0]
	def update_insects():
		nonlocal pos, orient
		diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]   # (N, N, 2)
		Dists = np.sqrt((diff ** 2).sum(axis=2))                 # (N, N)
		np.fill_diagonal(Dists, np.inf)
		WW = 1.0 / (Dists ** gamma)
		WW /= WW.sum(axis=1, keepdims=True)
		mf_pos = WW @ pos
		vx, vy = np.cos(orient), np.sin(orient)
		Dx, Dy = mf_pos[:, 0] - pos[:, 0], mf_pos[:, 1] - pos[:, 1]
		dtheta = np.pi*(Dy*vx - Dx*vy)/L     #scalar regjection of D onto v
		orient += np.clip(dtheta, -np.pi, np.pi) + eta*np.random.uniform(-np.pi, np.pi, size=N) #ensures determinisitc part of turn <= pi
		cos_orient, sin_orient = np.cos(orient), np.sin(orient)
		pos[:, 0] += cos_orient
		pos[:, 1] += sin_orient
		return cos_orient, sin_orient
	return update_insects


custom_cmap = mcolors.LinearSegmentedColormap.from_list('bog', ["#00b3ff", "#ff0000", "#FFFF00"], N=15)  #N=256, 15


N      =   20     #number of insects   (almost 4 hours for N=20, 30X30 1 trial)
num_trials = 31     #use odd number, takes median


Gammas = np.linspace(0, 6, 30)  #41
Etas   = np.linspace(0, 1, 30)  #41


#np.random.seed(44)
time_steps =   10**5   #100k
fit_start_t = int(time_steps/4)  #cutting out first 1/4
L          =   100     #length scale of rotational acceleration  #100
Powers = np.zeros((len(Gammas), len(Etas)))
for i in range(len(Gammas)):
	for j in range(len(Etas)):
		gamma, eta = Gammas[i], Etas[j]
		Slopes = np.zeros(num_trials)
		for trial in range(num_trials):
			#Scale and Initial Conditions Parameters
			Angs = np.random.uniform(-np.pi, np.pi, size=N)
			Rs = np.random.uniform(0, 1, size=N)**0.5 * (L/2)   #initializing positions on a disk
			pos = np.zeros((N, 2))
			pos[:, 0], pos[:, 1] = Rs*np.cos(Angs), Rs*np.sin(Angs)
			orient = np.random.uniform(-np.pi, np.pi, size=N)
			update_insects = make_updater_function(pos, orient, L, gamma, eta)  #making updater function
			#calculating statistics
			r_var = np.zeros(time_steps)
			for k in range(time_steps):
				update_insects()
				r_var[k] = pos[:, 0].var() + pos[:, 1].var()
			tt = np.linspace(1, time_steps, time_steps)
			log_r_var = np.log(r_var)
			log_t = np.log(tt)
			slope, intercept = np.polyfit(log_t[fit_start_t:], log_r_var[fit_start_t:], 1)
			Slopes[trial] = slope
		print(i, np.round(gamma, 4), np.round(eta, 4), np.round(np.median(Slopes), 6))
		Powers[i, j] = np.median(Slopes)


fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)   #cmaps: brg, tab10
im = ax.imshow(Powers.T, cmap=custom_cmap, origin='lower', extent=[ Gammas[0], Gammas[-1], Etas[0], Etas[-1] ], aspect='auto', vmin=0, vmax=2.0)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"Scaling exponent")
ax.set_xlabel(r"Mean field weight exponent ($\gamma$)")
ax.set_ylabel(r"Noise strength ($\eta$)")



np.savez(r"C:\Users\Charles\Desktop\Justin_Data_N=20.npz", Gammas=Gammas, Etas=Etas, Powers=Powers)



#plt.savefig("C:/Users/Justin/Desktop/Swarm_stability.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after








