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



#PANEL 1 - SHOWING WHOLE SWARM
gamma  =  2      #2.41   #accoustic weighting  1/r^gamma
eta    =  0.1     #0.0847   #noise strength (0 to 1)
N1     =  100       #number of insects
L      =  100        #upper bound length scale (100)

np.random.seed(14)
time_steps = 100
time_steps_cut = 1000

#Scale and Initial Conditions Parameters
Angs = np.random.uniform(-np.pi, np.pi, size=N1)
Rs = np.random.uniform(0, 1, size=N1)**0.5 * (L/2)   #initializing positions on a disk
pos = np.zeros((N1, 2))
pos[:, 0], pos[:, 1] = Rs*np.cos(Angs), Rs*np.sin(Angs)
orient = np.random.uniform(-np.pi, np.pi, size=N1)
update_insects = make_updater_function(pos, orient, L, gamma, eta)  #making updater function

Pos1 = np.zeros((time_steps, N1, 2))
Orient1 = np.zeros((time_steps, N1))
for i in range(time_steps_cut):
	update_insects()
for i in range(time_steps):
	update_insects()
	Pos1[i, :, :] = pos
	Orient1[i, :] = orient






#PANEL 2 SHOWING PAIR FORMATION
gamma  =  3      #2.41   #accoustic weighting  1/r^gamma
eta    =  0.0     #0.0847   #noise strength (0 to 1)
N2     =  30       #number of insects

np.random.seed(30)   #seed23 with N2=50   #seed21 with N2=30  #seed30 with N2=30
time_steps = 100
time_steps_cut = 100

#Scale and Initial Conditions Parameters
Angs = np.random.uniform(-np.pi, np.pi, size=N2)
Rs = np.random.uniform(0, 1, size=N2)**0.5 * (L/2)   #initializing positions on a disk
pos = np.zeros((N2, 2))
pos[:, 0], pos[:, 1] = Rs*np.cos(Angs), Rs*np.sin(Angs)
orient = np.random.uniform(-np.pi, np.pi, size=N2)
update_insects = make_updater_function(pos, orient, L, gamma, eta)  #making updater function

Pos2 = np.zeros((time_steps, N2, 2))
Orient2 = np.zeros((time_steps, N2))
for i in range(time_steps_cut):
	update_insects()
for i in range(time_steps):
	update_insects()
	Pos2[i, :, :] = pos
	Orient2[i, :] = orient






#PANEL 3 Density Functions  (SLOW PART HERE)
Gammas = [0, 1, 2]
eta    =  0.1     #0.0847   #noise strength (0 to 1)
N3     =  100       #number of insects

np.random.seed(42)  
time_steps     = 30000   #SLOW PART HERE  - use 30k
time_steps_cut = 1000

#Scale and Initial Conditions Parameters
Angs = np.random.uniform(-np.pi, np.pi, size=N3)
Rs = np.random.uniform(0, 1, size=N3)**0.5 * (L/2)   #initializing positions on a disk
pos = np.zeros((N3, 2))
pos[:, 0], pos[:, 1] = Rs*np.cos(Angs), Rs*np.sin(Angs)
orient = np.random.uniform(-np.pi, np.pi, size=N3)

RS = np.zeros((len(Gammas), time_steps*N3))
for g in range(len(Gammas)):
	gamma = Gammas[g]
	update_insects = make_updater_function(pos, orient, L, gamma, eta)  #making updater function
	Radius = np.zeros((time_steps, N3))
	for i in range(time_steps_cut):
		update_insects()
	for i in range(time_steps):
		update_insects()
		x_mean, y_mean = pos[:, 0].mean(), pos[:, 1].mean()
		Radius[i, :] = np.sqrt( (pos[:, 0] - x_mean)**2 + (pos[:, 1] - y_mean)**2  )
	RS[g, :] = Radius.flatten()







fig = plt.figure(figsize=(6, 6), constrained_layout=True)
#plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0.5, wspace=0.5)
ax0  = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax1  = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax2  = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax3  = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)



#PANEL 0 - GAMMA ILLUSTRATION
vector_clrs = ["#0000FF", "#00C76A", "#FF00BF"]
gamma1, gamma2, gamma3 = 0, 2, 4
ref_insect = np.array([3, 2])
neighbors = np.array([ [2.5, 4.5],  [6, 8],  [7.5, 7.3], [8.5, 7.2], [9, 8], [8.3, 9], [7.1, 8.4], [8, 8.2]])
diffs = neighbors - ref_insect
dists = np.sqrt((diffs ** 2).sum(axis=1))
W1 = 1.0 / (dists ** gamma1)
W2 = 1.0 / (dists ** gamma2)
W3 = 1.0 / (dists ** gamma3)
mf_pos1 = (W1/W1.sum()) @ neighbors
mf_pos2 = (W2/W2.sum()) @ neighbors
mf_pos3 = (W3/W3.sum()) @ neighbors
ax0.plot(ref_insect[0], ref_insect[1], "o", color="#FF0000", zorder=1)
ax0.plot(neighbors[:, 0], neighbors[:, 1], "o", color="#00B7FF")
ax0.quiver(0, 0,  ref_insect[0], ref_insect[1], angles='xy', scale_units='xy', scale=1, color='black', zorder=10)
ax0.quiver(ref_insect[0], ref_insect[1],  mf_pos1[0]-ref_insect[0], mf_pos1[1]-ref_insect[1], angles='xy', scale_units='xy', scale=1, color=vector_clrs[0], zorder=10)
ax0.quiver(ref_insect[0], ref_insect[1],  mf_pos2[0]-ref_insect[0], mf_pos2[1]-ref_insect[1], angles='xy', scale_units='xy', scale=1, color=vector_clrs[1], zorder=10)
ax0.quiver(ref_insect[0], ref_insect[1],  mf_pos3[0]-ref_insect[0], mf_pos3[1]-ref_insect[1], angles='xy', scale_units='xy', scale=1, color=vector_clrs[2], zorder=10)
ax0.set_xlim(0, 10)
ax0.set_ylim(0, 10)
ax0.set_xticks([0])
ax0.set_yticks([0])
ax0.text(1.8, 0.4, r"$\vec{r_i}$", color='black', fontsize=11)
ax0.text(6, 5, r"$\vec{D_i}$ ($\gamma=0$)", color=vector_clrs[0], fontsize=11)
ax0.text(3, 6.5, r"$\vec{D_i}$ ($\gamma=2$)", color=vector_clrs[1], fontsize=11)
ax0.text(0.5, 5.2, r"$\vec{D_i}$ ($\gamma=4$)", color=vector_clrs[2], fontsize=11)




#PANEL 1
custom_grey = "#707070"
x_mf1, y_mf1 = Pos1[-1, :, 0].mean(), Pos1[-1, :, 1].mean()
Pos1[:, :, 0] -= x_mf1
Pos1[:, :, 1] -= y_mf1
for j in range(N1):
	ax1.plot(Pos1[:, j, 0], Pos1[:, j, 1], "-", color=custom_grey, linewidth=0.1, zorder=1)
ax1.quiver(Pos1[-1, :, 0], Pos1[-1, :, 1], np.cos(Orient1[-1, :]), np.sin(Orient1[-1, :]), Orient1[-1, :], clim=[-np.pi, np.pi], cmap=mcolors.ListedColormap(["#000000"]), zorder=10)
panel1_limit = L/2
ax1.set_xticks([-L, -L/2, 0, L/2, L], labels=[r"$-L$", r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$", r"$L$"], fontsize=12)
ax1.set_yticks([-L, -L/2, 0, L/2, L], labels=[r"$-L$", r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$", r"$L$"], fontsize=12)
ax1.set_xlim(-panel1_limit, panel1_limit)
ax1.set_ylim(-panel1_limit, panel1_limit)


#PANEL 2
clrs = ["#007BFF", "#FF0000"]
x_mf2, y_mf2 = Pos2[-1, :, 0].mean(), Pos2[-1, :, 1].mean()
Pos2[:, :, 0] -= x_mf2
Pos2[:, :, 1] -= y_mf2
for j in range(N2):
	ax2.plot(Pos2[:, j, 0], Pos2[:, j, 1], "-", color=custom_grey, linewidth=0.1, zorder=1)
ax2.quiver(Pos2[-1, :, 0], Pos2[-1, :, 1], np.cos(Orient2[-1, :]), np.sin(Orient2[-1, :]), Orient2[-1, :], clim=[-np.pi, np.pi], cmap=mcolors.ListedColormap(["#000000"]), zorder=10)
ax2.set_xticks([-L, 0, L], labels=[r"$-L$", r"$0$", r"$L$"], fontsize=12)
ax2.set_yticks([-L, 0, L], labels=[r"$-L$", r"$0$", r"$L$"], fontsize=12)
ax2.set_xlim(-0.5*L, L)
ax2.set_ylim(-L, 0.5*L)
pair_member1, pair_member2 = 4, 24   #4, 24
ax2.plot(Pos2[:, pair_member1, 0], Pos2[:, pair_member1, 1], "-", color=clrs[0], linewidth=1, zorder=5)
ax2.plot(Pos2[:, pair_member2, 0], Pos2[:, pair_member2, 1], "-", color=clrs[1], linewidth=1, zorder=5)


#PANEL 3
hist_clrs = ["#0000FF", "#FF6F00", "#00C76A"]
r_bin_edges = np.linspace(0, 50, 51)
bin_width = r_bin_edges[1] - r_bin_edges[0]
r_bin_cents = r_bin_edges[:-1] + bin_width/2
Areas = np.pi*(r_bin_edges[1:]**2 - r_bin_edges[:-1]**2)
Counts1 = np.histogram(RS[0, :], bins=r_bin_edges)[0]
Counts2 = np.histogram(RS[1, :], bins=r_bin_edges)[0]
Counts3 = np.histogram(RS[2, :], bins=r_bin_edges)[0]
rho1 = Counts1/(Areas*time_steps)
rho2 = Counts2/(Areas*time_steps)
rho3 = Counts3/(Areas*time_steps)
ax3.plot(r_bin_cents, rho1, color=hist_clrs[0], label=r"$\gamma=0$")
ax3.plot(r_bin_cents, rho2, color=hist_clrs[1], label=r"$\gamma=1$")
ax3.plot(r_bin_cents, rho3, color=hist_clrs[2], label=r"$\gamma=2$")
ax3.set_xlim(0, L/2)
ax3.set_xticks([0, L/4, L/2], labels=[r"$0$", r"$\frac{L}{4}$", r"$\frac{L}{2}$"], fontsize=12)
ax3.set_ylim(0, 0.7)
ax3.set_xlabel("Distance from center")
ax3.set_ylabel("Density (insects/area)")
ax3.legend()


ax0.text(0.02, 0.98, "(a)", fontsize=11, transform=ax0.transAxes, va='top')
ax1.text(0.02, 0.98, "(b)", fontsize=11, transform=ax1.transAxes, va='top')
ax2.text(0.02, 0.98, "(c)", fontsize=11, transform=ax2.transAxes, va='top')
ax3.text(0.02, 0.98, "(d)", fontsize=11, transform=ax3.transAxes, va='top')


plt.savefig("C:/Users/Justin/Desktop/Fig1.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after







