import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import code


#creates function used for updating positions/orientions of all swarm memebers
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



np.random.seed(57)   #57


time_steps =   1 * 10**5   #100k
L          =   100     #length scale of rotational acceleration  #100
N          =   100     #number of insects
Gammas     = [1.0, 2.0, 2.5, 3.0]   #accoustic weight  1/r^gamma
Etas       = [0.0, 0.1, 1.0]         #noise strength (0 to 1)


#clrs  =["#ff1f5b", "#00cd6c", "#009ade" ,"#af58ba", "#f28522"]
clrs  =["#00cd6c" ,"#af58ba", "#f28522"]

fig = plt.figure(figsize=(5, 5), constrained_layout=True)
#plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0.5, wspace=0.5)
ax1  = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax2  = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax3  = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax4  = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
axs = [ax1, ax2, ax3, ax4]


for pwr in range(len(Gammas)):
	for nz in range(len(Etas)):
		print(pwr, nz)
		gamma, eta, ax = Gammas[pwr], Etas[nz], axs[pwr]

		#Scale and Initial Conditions Parameters
		Angs = np.random.uniform(-np.pi, np.pi, size=N)
		Rs = np.random.uniform(0, 1, size=N)**0.5 * (L/2)   #initializing positions on a disk
		pos = np.zeros((N, 2))
		pos[:, 0], pos[:, 1] = Rs*np.cos(Angs), Rs*np.sin(Angs)
		orient = np.random.uniform(-np.pi, np.pi, size=N)
		update_insects = make_updater_function(pos, orient, L, gamma, eta)  #making updater function

		#calculating statistics
		r_var = np.zeros(time_steps)
		for i in range(time_steps):
			update_insects()
			r_var[i] = pos[:, 0].var() + pos[:, 1].var()
		tt = np.linspace(1, time_steps+1, time_steps)
		ax.plot(tt, r_var, color=clrs[nz], label=r"$\eta=$" + str(eta))

	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_title(r'$\gamma =$' + str(gamma))
	t0, x0 = 100, 1000  #start pt for power laws
	ax.plot(tt[t0:], x0*(tt[t0:]/t0)**1, ":", color='red')
	ax.plot(tt[t0:], x0*(tt[t0:]/t0)**2, ":", color='black')
	ax.set_xticks([1, 10, 100, 1000, 10000, 100000, 1000000])
	ax.set_xlim(1, 1.5*time_steps)
	ax.set_ylim(10, 1e10)

ax1.set_xlabel("Time steps")
ax2.set_xlabel("Time steps")
ax3.set_xlabel("Time steps")
ax4.set_xlabel("Time steps")
#ax1.set_ylabel(r'$\langle r(t)^2 \rangle$')
ax1.set_ylabel("Variance")
ax2.set_ylabel("Variance")
ax3.set_ylabel("Variance")
ax4.set_ylabel("Variance")
ax1.legend()



plt.savefig("C:/Users/Justin/Desktop/Swarm_stability.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after





