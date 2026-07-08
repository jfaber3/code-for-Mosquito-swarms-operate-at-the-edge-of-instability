import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation, FFMpegWriter
import code
import time


#creates function used for updating positions/orientions of all swarm memebers

def make_updater_function1(pos, orient, L, gamma, eta):
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


def make_updater_function2(pos, orient, L, gamma, eta):
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




#Model Parameters
gamma  =  1      #2.41   #accoustic weighting  1/r^gamma
eta    =  0.1     #0.0847   #noise strength (0 to 1)
N      =  100       #number of insects
L      =  100        #upper bound length scale (100)


num_trials = 100
time_steps = 100

time1 = time.time()

for trial in range(num_trials):
	print(trial)
	#Scale and Initial Conditions Parameters
	Angs = np.random.uniform(-np.pi, np.pi, size=N)
	Rs = np.random.uniform(0, 1, size=N)**0.5 * (L/2)   #initializing positions on a disk
	pos = np.zeros((N, 2))
	pos[:, 0], pos[:, 1] = Rs*np.cos(Angs), Rs*np.sin(Angs)
	orient = np.random.uniform(-np.pi, np.pi, size=N)
	
	update_insects = make_updater_function2(pos, orient, L, gamma, eta)  #making updater function

	X, Y = np.zeros(time_steps), np.zeros(time_steps)
	for i in range(time_steps):
		update_insects()
		X[i], Y[i] = pos[:, 0].mean(), pos[:, 1].mean()

print(np.round(time.time() - time1, 4))





'''
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
xlimit = 1*L  #limits of animation plot
num_frames = 100000 
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-xlimit, xlimit)
ax.set_ylim(-xlimit, xlimit)
qv = ax.quiver(pos[:, 0], pos[:, 1], np.cos(orient), np.sin(orient), orient, clim=[-np.pi, np.pi], cmap=mcolors.ListedColormap(['black']))
cm = ax.plot(pos[:, 0].mean(), pos[:, 1].mean(), "o", color='dodgerblue', markersize=3)
anim = FuncAnimation(fig, Animate, frames=num_frames, interval=0.001, repeat=False, blit=False)
'''



#plt.savefig("C:/Users/Justin/Desktop/Swarm_stability.pdf", dpi=300)
#plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after










