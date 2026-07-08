import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation, FFMpegWriter
import code


#creates function used for updating positions/orientions of all swarm memebers
def make_updater_function(pos, orient, L, gamma, eta):
	N = pos.shape[0]
	Boundary = L
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

		# Reflective boundary conditions
		hit_x = (pos[:, 0] < -Boundary ) | (pos[:, 0] > Boundary )
		hit_y = (pos[:, 1] < -Boundary ) | (pos[:, 1] > Boundary )
		orient = np.where(hit_x, np.pi - orient, orient)  # flip x-component
		orient = np.where(hit_y, -orient, orient)         # flip y-component
		return cos_orient, sin_orient
	return update_insects



#Model Parameters
gamma  =  3      #2.41   #accoustic weighting  1/r^gamma
eta    =  0.1     #0.0847   #noise strength (0 to 1)
N      =  100       #number of insects
L      =  100        #upper bound length scale (100)


#Scale and Initial Conditions Parameters
Angs = np.random.uniform(-np.pi, np.pi, size=N)
Rs = np.random.uniform(0, 1, size=N)**0.5 * (L/2)   #initializing positions on a disk
pos = np.zeros((N, 2))
pos[:, 0], pos[:, 1] = Rs*np.cos(Angs), Rs*np.sin(Angs)
orient = np.random.uniform(-np.pi, np.pi, size=N)
update_insects = make_updater_function(pos, orient, L, gamma, eta)  #making updater function


#ANIMATION CODE
def Animate(i):
	if i%100 == 0:
		print(i)
	cos_orient, sin_orient = update_insects()
	ax.plot(pos[:, 0].mean(), pos[:, 1].mean(), "o", color='dodgerblue', markersize=3)
	qv.set_offsets(pos) #updating position on plot
	qv.set_UVC(cos_orient, sin_orient, orient) #updating orientation on plot
	return qv, ax

#setting up animation
xlimit = 1.1*L  #limits of animation plot
num_frames = 10000
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-xlimit, xlimit)
ax.set_ylim(-xlimit, xlimit)
qv = ax.quiver(pos[:, 0], pos[:, 1], np.cos(orient), np.sin(orient), orient, clim=[-np.pi, np.pi], cmap=mcolors.ListedColormap(['black']))
cm = ax.plot(pos[:, 0].mean(), pos[:, 1].mean(), "o", color='dodgerblue', markersize=3)
anim = FuncAnimation(fig, Animate, frames=num_frames, interval=2, repeat=False, blit=False)


#plt.savefig("C:/Users/Justin/Desktop/Swarm_stability.pdf", dpi=300)
plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after







