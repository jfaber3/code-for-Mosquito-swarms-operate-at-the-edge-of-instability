import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.ndimage import uniform_filter, gaussian_filter
import matplotlib.colors as mcolors
import code
from scipy.optimize import curve_fit


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




clrs  =["#1100ff", "#00d5ff", "#00ff62", "#ffaa00", "#ff0000", "#ae00ff"]

time_steps     = 10**4   #100k or 10k
time_steps_cut = 10**3   #steps to add then cut out
#L      =   100     #length scale of rotational acceleration  #100


N      =  100
gamma  =  0.0
eta    =  0.5

ss = []
l_values = [5, 10, 25, 50, 75, 100, 125, 150, 175, 200]

for L in l_values:
	#Scale and Initial Conditions Parameters
	Angs = np.random.uniform(-np.pi, np.pi, size=N)
	Rs = np.random.uniform(0, 1, size=N)**0.5 * (L/2)   #initializing positions on a disk
	pos = np.zeros((N, 2))
	pos[:, 0], pos[:, 1] = Rs*np.cos(Angs), Rs*np.sin(Angs)
	orient = np.random.uniform(-np.pi, np.pi, size=N)
	update_insects = make_updater_function(pos, orient, L, gamma, eta)  #making updater function

	print(Rs)

	#calculating statistics
	Rs = []
	swarm_sizes = []
	Ns = []


	for i in range(time_steps + time_steps_cut):
		update_insects()
		x_mean, y_mean = pos[:, 0].mean(), pos[:, 1].mean() #use x mean and y mean for r mean
		R = np.sqrt(    (pos[:, 0]-x_mean)**2  +  (pos[:, 1]-y_mean)**2    ) 


		Rs.extend(R)
		swarm_sizes.append( R.mean() ) 

	#Rs = np.array(Rs[N*time_steps_cut:]) # all the Ri of every particle
	#everything in 1 dimension

	print(len(Rs))
	print(Rs)

	ss.append(np.mean(swarm_sizes[time_steps_cut:]))



print("ss", ss)


plt.plot(l_values, ss, "o-")



plt.xlabel("$L$")
plt.ylabel("Mean Swarm Size")
#plt.title("Mean Swarm Size vs L")
plt.savefig("/Users/annabelleboots/Desktop/mss_L.pdf", dpi=300)

plt.show()


#plt.plot(swarm_sizes)
#plt.axvline(x=time_steps_cut)

plt.show()


code.interact(local=locals())  #allows interaction with variables in terminal after





