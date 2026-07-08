import numpy as np
import code
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


DATA1, start_frame1, stop_frame1 = np.load("C:/Users/Justin/Desktop/Swarming/Mosquito Swarm Data/Poda-Swarm-1.npy"), 46000, 52500
DATA2, start_frame2, stop_frame2 = np.load("C:/Users/Justin/Desktop/Swarming/Mosquito Swarm Data/Poda-Swarm-2.npy"), 46000, 52500
DATA3, start_frame3, stop_frame3 = np.load("C:/Users/Justin/Desktop/Swarming/Mosquito Swarm Data/Poda-Swarm-3.npy"), 46000, 52500
DATA4, start_frame4, stop_frame4 = np.load("C:/Users/Justin/Desktop/Swarming/Mosquito Swarm Data/Poda-Swarm-4.npy"), 46000, 50000
DATA5, start_frame5, stop_frame5 = np.load("C:/Users/Justin/Desktop/Swarming/Mosquito Swarm Data/Poda-Swarm-5.npy"), 46000, 48000
DATA6, start_frame6, stop_frame6 = np.load("C:/Users/Justin/Desktop/Swarming/Mosquito Swarm Data/Poda-Swarm-6.npy"), 46000, 52500

Data1 = DATA1[:,  start_frame1:stop_frame1,  :]
Data2 = DATA2[:,  start_frame2:stop_frame2,  :]
Data3 = DATA3[:,  start_frame3:stop_frame3,  :]
Data4 = DATA4[:,  start_frame4:stop_frame4,  :]
Data5 = DATA5[:,  start_frame5:stop_frame5,  :]
Data6 = DATA6[:,  start_frame6:stop_frame6,  :]

Data1[np.isnan(Data1)] = 0  #set nans to 0
Data2[np.isnan(Data2)] = 0
Data3[np.isnan(Data3)] = 0
Data4[np.isnan(Data4)] = 0
Data5[np.isnan(Data5)] = 0
Data6[np.isnan(Data6)] = 0

dt = 0.02  #50 frames/sec




def find_CM(Data):  #calculates mean field of any pos, vel, or acc
    CM = np.zeros((Data.shape[1], 3))
    mask = np.linalg.norm(Data, axis=2) != 0
    valid_counts = mask.sum(axis=0)
    valid_counts = np.where(valid_counts == 0, 1, valid_counts)
    CM = (Data * mask[..., None]).sum(axis=0) / valid_counts[:, None]
    return CM

def make_vel_mask(Data, window_length): #mask to cut out all unusable data, near edges and near missing segments
    half_win = window_length//2
    N, T = Data.shape[:2]
    mask = np.linalg.norm(Data, axis=2) != 0
    vel_mask = mask.copy()
    for i in range(2, T-2):
        for j in range(N):
            if sum(mask[j, i-half_win : i+half_win+1]) < window_length:   #getting rid of points without half_win neighbors on each side
                vel_mask[j, i] = False
    vel_mask[:, :half_win], vel_mask[:, -half_win:] = False, False  #getting rid of end points in time
    return vel_mask

def Savitzky_Golay_Filter(Data, dt, window_length, polyorder):   #use odd window length (11) and polyorder=3.  Data shape: (N_particles, T, 3)
    N = Data.shape[0]
    pos_smooth = np.zeros_like(Data)
    vel_smooth = np.zeros_like(Data)
    acc_smooth = np.zeros_like(Data)
    for i in range(N):
        for k in range(3):
            x = Data[i, :, k]
            pos_smooth[i, :, k] = savgol_filter(x, window_length, polyorder, deriv=0, delta=dt, mode='interp')
            vel_smooth[i, :, k] = savgol_filter(x, window_length, polyorder, deriv=1, delta=dt, mode='interp')
            acc_smooth[i, :, k] = savgol_filter(x, window_length, polyorder, deriv=2, delta=dt, mode='interp')
    return pos_smooth, vel_smooth, acc_smooth

def get_all_data(Data, window_length, polyorder=3):
    data_mask = make_vel_mask(Data, window_length)
    pos_data, vel_data, acc_data = Savitzky_Golay_Filter(Data, dt, window_length, polyorder)
    pos_data = pos_data * data_mask[..., None]
    vel_data = vel_data * data_mask[..., None]
    acc_data = acc_data * data_mask[..., None]
    pos_CM, vel_CM, acc_CM = find_CM(pos_data), find_CM(vel_data), find_CM(acc_data)
    return pos_data, vel_data, acc_data, pos_CM, vel_CM, acc_CM, data_mask


clrs  =["#1100ff", "#00d5ff", "#00ff62", "#ffaa00", "#ff0000", "#ae00ff"]
clrs2  =["#005eff", "#ff0000", "#a600ff"]



window_length = 15   #use 15  (min 5)
filt_order    = 3    #use 3   (min 2, max = window-1)





def Map_Data(Data, ax, swarm_number):

    #FINDING Diffusion Constant from EXPERIMENTAL DATA

    pos_data, vel_data, _, pos_CM, _, _, data_mask = get_all_data(Data, window_length, polyorder=filt_order)
    T = vel_data.shape[1]


    # --- Compute unit velocity vectors, masking invalid frames ---
    speed = np.linalg.norm(vel_data, axis=2)  # (N, T)
    valid = data_mask & (speed > 1e-6)        # avoid division by zero

    #finding mean speed and radius for useful units
    mean_speed = speed[valid].mean()
    Rs = np.linalg.norm(pos_data - pos_CM, axis=2)
    mean_radius = np.mean(Rs[valid])

    # Unit vectors; set invalid frames to zero
    u = np.zeros_like(vel_data)
    u[valid] = vel_data[valid] / speed[valid, None]

    #AMSD = <arccos(u(t)·u(t+tau))^2>, use tau = 1
    # Both endpoints must be valid
    valid_pair = valid[:, 1:] & valid[:, :T-1]   # (N, T-tau)

    # Dot product of unit vectors
    dot = np.sum(u[:, 1:, :] * u[:, :T-1, :], axis=2)  # (N, T-tau)
    dot = np.clip(dot, -1.0, 1.0)    # numerical safety for arccos
    angle = np.arccos(dot)           # (N, T-tau), in radians

    # Only average over valid pairs
    n = valid_pair.sum()
    AMSD = np.sum(angle[valid_pair]**2) / n
    Dr = (1/2) * AMSD/dt   #rad^2 / sec
    T_char = mean_radius/mean_speed  #sec
    DeffT_data = Dr*T_char  #rad^2

    print(f"Dr from data:            {Dr:.4f} rad²/sec")
    print(f"T_char from data:        {T_char:.4f} sec")
    print(f"Deff_T from data:        {DeffT_data:.4f} rad^2")

    #FINDING Correlation Length from Experimental Data

    cor_steps = 30
    C      = np.zeros(cor_steps)
    for tau in range(cor_steps):
        valid_pair = valid[:, tau:] & valid[:, :T-tau]
        dot = np.sum(u[:, tau:, :] * u[:, :T-tau, :], axis=2)
        dot = np.clip(dot, -1.0, 1.0)
        n = valid_pair.sum()
        if n > 0:
            C[tau] = np.sum(dot[valid_pair]) / n

    for tau in range(cor_steps):
        if C[tau] < 1/np.e:
            c1, c2, t1, t2 = C[tau-1] - 1/np.e,  C[tau] - 1/np.e,  tau-1,  tau  #now find intersection with x-axis
            cor_time = dt * (t1 - c1*( (t2-t1)/(c2-c1) ))  #sec
            break

    cor_length = cor_time*mean_speed
    Coherence_data = cor_length/mean_radius

    print(f"Corr Time from data:        {cor_time:.4f} sec")
    print(f"Corr Len from data:        {cor_length:.4f} m")
    print(f"Coherence from data:        {Coherence_data:.4f}")


    '''
    #plot correlation function
    plt.figure()
    plt.plot(np.arange(cor_steps) * dt, C, "o-", color='black')
    plt.axhline(y=1/np.e, color='red')
    plt.xlabel('τ (s)')
    plt.ylabel('C(τ)')
    plt.title('Direction Autocorrelation')
    plt.show()
    '''


    #Finding all points within some fraction of the experimental values
    tol_C = 0.1
    tol_D = 0.1
    Gammas_all, Etas_all   = [], []
    Gammas_C, Etas_C       = [], []
    Gammas_D, Etas_D       = [], []
    Gammas_best, Etas_best = [], []
    for i in range(len(Gammas0)):
        for j in range(len(Etas0)):
            Gammas_all.append(Gammas0[i])
            Etas_all.append(Etas0[j])
            C_match = abs(Coherence[j, i] - Coherence_data)   / Coherence_data < tol_C
            D_match = abs(Deff_T[j, i]    - DeffT_data)   / DeffT_data     < tol_D
            if C_match:
                Gammas_C.append(Gammas0[i])
                Etas_C.append(Etas0[j])
            if D_match:
                Gammas_D.append(Gammas0[i])
                Etas_D.append(Etas0[j])
            if C_match and D_match:
                Gammas_best.append(Gammas0[i])
                Etas_best.append(Etas0[j])


    ax.set_title(r"Swarm " + str(swarm_number) + r":  $\gamma_{fit} \approx $" + str(np.round(np.mean(Gammas_best), 2)) + r"  $\eta_{fit} \approx $" + str(np.round(np.mean(Etas_best), 3)), fontsize=9)
    ax.scatter(Gammas_all, Etas_all, color='grey',s=0.2)
    ax.scatter(Gammas_C, Etas_C,     color=clrs2[0], s=3, alpha=0.5)
    ax.scatter(Gammas_D, Etas_D,     color=clrs2[1], s=3, alpha=0.5)
    ax.scatter(Gammas_best, Etas_best,  color=clrs2[2], s=10)




#LOADING SIMULATION DATA

Data_sim = np.load(r"C:\Users\Justin\Desktop\Swarming\Manuscript_Figs\2D_model_R0_CorLen_data_N=20.npz")
Gammas0, Etas0, Swarm_Sizes0, Cor_Lens0 = Data_sim["Gammas"], Data_sim["Etas"], Data_sim["Swarm_Sizes"], Data_sim["Cor_Lens"]
X, Y = np.meshgrid(Gammas0, Etas0)
Z1 = np.median(Swarm_Sizes0, axis=2).T
Z2 = np.median(Cor_Lens0, axis=2).T
Coherence = Z2/Z1
Deff_T = (1/6) * (np.pi**2) * (Y**2) * Z1   #effective diffusion constant times characteristic time (time to fly R0) (l=1) (number of time steps to fly R0)



fig = plt.figure(figsize=(5, 7), constrained_layout=True)
#plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0.5, wspace=0.5)
ax1  = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1)
ax2  = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1)
ax3  = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1)
ax4  = plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1)
ax5  = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1)
ax6  = plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1)



Map_Data(Data1, ax1, 1)
Map_Data(Data2, ax2, 2)
Map_Data(Data3, ax3, 3)
Map_Data(Data4, ax4, 4)
Map_Data(Data5, ax5, 5)
Map_Data(Data6, ax6, 6)


ax5.set_xlabel(r"$\gamma$")
ax6.set_xlabel(r"$\gamma$")
ax1.set_ylabel(r"$\eta$")
ax3.set_ylabel(r"$\eta$")
ax5.set_ylabel(r"$\eta$")



plt.savefig("C:/Users/Justin/Desktop/Data_Mapping.pdf", dpi=300)





#Plotting just SIMULATION DATA 3D Plots
fig, axes = plt.subplots(2, 2, figsize=(8.5, 7), subplot_kw={"projection": "3d"})
ax1, ax2 = axes[0]
ax3, ax4 = axes[1]

# --- Panel 1: Swarm Radius ---
surf1 = ax1.plot_surface(X, Y, Z1, cmap='jet', linewidth=0, antialiased=False, vmax=50)
ax1.set_xlabel(r"$\gamma$")
ax1.set_ylabel(r"$\eta$")
ax1.set_zlabel("Swarm Radius")
ax1.view_init(elev=30, azim=150)
ax1.set_box_aspect((1, 1, 1.1))
ax1.set_zlim(0, 50)

# --- Panel 2: Correlation Length ---
surf2 = ax2.plot_surface(X, Y, Z2, cmap='jet', linewidth=0, antialiased=False)
ax2.set_xlabel(r"$\gamma$")
ax2.set_ylabel(r"$\eta$")
ax2.set_zlabel("Correlation Length")
ax2.view_init(elev=30, azim=150)
ax2.set_box_aspect((1, 1, 1.1))

# --- Panel 3: Coherence Ratio ---
surf3 = ax3.plot_surface(X, Y, Coherence, cmap='jet', linewidth=0, antialiased=False)
ax3.set_xlabel(r"$\gamma$")
ax3.set_ylabel(r"$\eta$")
ax3.set_zlabel(r"$R_{cor}/R_{swarm}$")
ax3.view_init(elev=30, azim=45)
ax3.set_box_aspect((1, 1, 1.1))

# --- Panel 4: Deff * Tc ---
surf4 = ax4.plot_surface(X, Y, Deff_T, cmap='jet', linewidth=0, antialiased=False)
ax4.set_xlabel(r"$\gamma$")
ax4.set_ylabel(r"$\eta$")
ax4.set_zlabel(r"$D_{eff}T_c$")
ax4.view_init(elev=30, azim=215)
ax4.set_box_aspect((1, 1, 1.1))

# --- Panel labels ---
for ax, label in zip([ax1, ax2, ax3, ax4], ['(a)', '(b)', '(c)', '(d)']):
    ax.set_title(label, loc='left')
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.3)

plt.savefig("C:/Users/Justin/Desktop/3D_plots.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after







