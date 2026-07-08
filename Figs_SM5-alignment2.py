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


def Savitzky_Golay_Filter(Data, dt, window_length, polyorder=3):   #use odd window length (11-15) and polyorder=3.  Data shape: (N_particles, T, 3)
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


def get_all_data(Data, window_length):
    data_mask = make_vel_mask(Data, window_length)
    pos_data, vel_data, acc_data = Savitzky_Golay_Filter(Data, dt, window_length)
    pos_data = pos_data * data_mask[..., None]
    vel_data = vel_data * data_mask[..., None]
    acc_data = acc_data * data_mask[..., None]
    pos_CM, vel_CM, acc_CM = find_CM(pos_data), find_CM(vel_data), find_CM(acc_data)
    return pos_data, vel_data, acc_data, pos_CM, vel_CM, acc_CM, data_mask


def flatten_data(data, data_mask):
    data_x = data[:, :, 0].flatten()[data_mask.flatten()]
    data_y = data[:, :, 1].flatten()[data_mask.flatten()]
    data_z = data[:, :, 2].flatten()[data_mask.flatten()]
    mag_data = np.linalg.norm(data, axis=2).flatten()[data_mask.flatten()]
    return data_x, data_y, data_z, mag_data




window_length = 15   #11 or 15

pos_data1, vel_data1, acc_data1, pos_CM1, vel_CM1, acc_CM1, data_mask1 = get_all_data(Data1, window_length)
pos_data2, vel_data2, acc_data2, pos_CM2, vel_CM2, acc_CM2, data_mask2 = get_all_data(Data2, window_length)
pos_data3, vel_data3, acc_data3, pos_CM3, vel_CM3, acc_CM3, data_mask3 = get_all_data(Data3, window_length)
pos_data4, vel_data4, acc_data4, pos_CM4, vel_CM4, acc_CM4, data_mask4 = get_all_data(Data4, window_length)
pos_data5, vel_data5, acc_data5, pos_CM5, vel_CM5, acc_CM5, data_mask5 = get_all_data(Data5, window_length)
pos_data6, vel_data6, acc_data6, pos_CM6, vel_CM6, acc_CM6, data_mask6 = get_all_data(Data6, window_length)





def find_velocity_alignment(pos_data, vel_data, data_mask):
    print("Finding velocity alignment")
    T = pos_data.shape[1]
    Dist, Vel_dots = [], []
    for t in range(T):
        mask_t = data_mask[:, t]
        Pos = pos_data[mask_t, t]
        Vel = vel_data[mask_t, t]
        N = len(Pos)
        if N <= 1:
            continue
        D_vecs = Pos[None, :, :] - Pos[:, None, :]
        D_mag = np.linalg.norm(D_vecs, axis=2)
        np.fill_diagonal(D_mag, np.inf)
        sorted_idx = np.argsort(D_mag, axis=1)    #sort distances to get nearest neighbors
        nn_idx = sorted_idx[:, 0]                   # (N)  indices of 1 nearest neighbors for each point
        Vel_unit = Vel/np.linalg.norm(Vel, axis=1)[:, None]
        Dist.extend(D_mag[np.arange(N), nn_idx])
        Vel_dots.extend( np.sum( Vel_unit*Vel_unit[nn_idx], axis=1)  )
    return np.array(Dist), np.array(Vel_dots)


Dist1, Dots1 = find_velocity_alignment(pos_data1, vel_data1, data_mask1)
Dist2, Dots2 = find_velocity_alignment(pos_data2, vel_data2, data_mask2)
Dist3, Dots3 = find_velocity_alignment(pos_data3, vel_data3, data_mask3)
Dist4, Dots4 = find_velocity_alignment(pos_data4, vel_data4, data_mask4)
Dist5, Dots5 = find_velocity_alignment(pos_data5, vel_data5, data_mask5)
Dist6, Dots6 = find_velocity_alignment(pos_data6, vel_data6, data_mask6)






def plot_vel_dot_range(Dist, Acc_par, dmins, dmaxs, clr, num_hist_bins=51):
    vel_dot_a = Acc_par[ (Dist >= dmins[0]) & (Dist <= dmaxs[0]) ]
    vel_dot_b = Acc_par[ (Dist >= dmins[1]) & (Dist <= dmaxs[1]) ]
    vel_dot_c = Acc_par[ (Dist >= dmins[2]) & (Dist <= dmaxs[2]) ]
    vel_dot_d = Acc_par[ (Dist >= dmins[3]) & (Dist <= dmaxs[3]) ]
    vel_dot_e = Acc_par[ (Dist >= dmins[4]) & (Dist <= dmaxs[4]) ]
    vel_dot_f = Acc_par[ (Dist >= dmins[5]) & (Dist <= dmaxs[5]) ]
    ax1.hist(vel_dot_a, bins=num_hist_bins, range=(-1, 1), histtype='step', color=clr, density=True)
    ax2.hist(vel_dot_b, bins=num_hist_bins, range=(-1, 1), histtype='step', color=clr, density=True)
    ax3.hist(vel_dot_c, bins=num_hist_bins, range=(-1, 1), histtype='step', color=clr, density=True)
    ax4.hist(vel_dot_d, bins=num_hist_bins, range=(-1, 1), histtype='step', color=clr, density=True)
    ax5.hist(vel_dot_e, bins=num_hist_bins, range=(-1, 1), histtype='step', color=clr, density=True)
    ax6.hist(vel_dot_f, bins=num_hist_bins, range=(-1, 1), histtype='step', color=clr, density=True)
    ax1.axvline(x=vel_dot_a.mean(), ls="dotted", color=clr)
    ax2.axvline(x=vel_dot_b.mean(), ls="dotted", color=clr)
    ax3.axvline(x=vel_dot_c.mean(), ls="dotted", color=clr)
    ax4.axvline(x=vel_dot_d.mean(), ls="dotted", color=clr)
    ax5.axvline(x=vel_dot_e.mean(), ls="dotted", color=clr)
    ax6.axvline(x=vel_dot_f.mean(), ls="dotted", color=clr)
    print(len(vel_dot_a), len(vel_dot_b), len(vel_dot_c), len(vel_dot_d), len(vel_dot_e), len(vel_dot_f))



#Ranges of nearest neighbor distances to compute histograms of acceleration projected on nearest neighbor vector
Dmaxs = [0.30, 0.10, 0.05, 0.02, 0.02, 0.01]
Dmins = [0.10, 0.05, 0.02, 0.00, 0.01,  0.0]
Num_Bins = 20

clrs =[ "#000000", "#ff0000", "#d000ff", "#2b00ff", "#00b3ff", "#00bb5a", "#ffb700" ]
fig = plt.figure(figsize=(5, 7), constrained_layout=True)
#plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, hspace=0.5, wspace=0.5)
ax1  = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=1)
ax2  = plt.subplot2grid((3, 2), (0, 1), rowspan=1, colspan=1)
ax3  = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1)
ax4  = plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1)
ax5  = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1)
ax6  = plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1)

ax1.axvline(x=0, ls='dashed', color='black')
ax2.axvline(x=0, ls='dashed', color='black')
ax3.axvline(x=0, ls='dashed', color='black')
ax4.axvline(x=0, ls='dashed', color='black')
ax5.axvline(x=0, ls='dashed', color='black')
ax6.axvline(x=0, ls='dashed', color='black')

plot_vel_dot_range(Dist1, Dots1, Dmins, Dmaxs, clrs[1], num_hist_bins=Num_Bins)
plot_vel_dot_range(Dist2, Dots2, Dmins, Dmaxs, clrs[2], num_hist_bins=Num_Bins)
plot_vel_dot_range(Dist3, Dots3, Dmins, Dmaxs, clrs[3], num_hist_bins=Num_Bins)
plot_vel_dot_range(Dist4, Dots4, Dmins, Dmaxs, clrs[4], num_hist_bins=Num_Bins)
plot_vel_dot_range(Dist5, Dots5, Dmins, Dmaxs, clrs[5], num_hist_bins=Num_Bins)
plot_vel_dot_range(Dist6, Dots6, Dmins, Dmaxs, clrs[6], num_hist_bins=Num_Bins)

plt_xlimit = 1
plt_ylimit = 3
ax1.set_xlim(-plt_xlimit, plt_xlimit)
ax2.set_xlim(-plt_xlimit, plt_xlimit)
ax3.set_xlim(-plt_xlimit, plt_xlimit)
ax4.set_xlim(-plt_xlimit, plt_xlimit)
ax5.set_xlim(-plt_xlimit, plt_xlimit)
ax6.set_xlim(-plt_xlimit, plt_xlimit)
ax1.set_ylim(0, plt_ylimit)
ax2.set_ylim(0, plt_ylimit)
ax3.set_ylim(0, plt_ylimit)
ax4.set_ylim(0, plt_ylimit)
ax5.set_ylim(0, plt_ylimit)
ax6.set_ylim(0, plt_ylimit)

ax1.set_title("Distance range = " + str(int(100*Dmins[0])) + " - " + str(int(100*Dmaxs[0])) + " cm", fontsize=10, color='blue')
ax2.set_title("Distance range = " + str(int(100*Dmins[1])) + " - " + str(int(100*Dmaxs[1])) + " cm", fontsize=10, color='blue')
ax3.set_title("Distance range = " + str(int(100*Dmins[2])) + " - " + str(int(100*Dmaxs[2])) + " cm", fontsize=10, color='blue')
ax4.set_title("Distance range = " + str(int(100*Dmins[3])) + " - " + str(int(100*Dmaxs[3])) + " cm", fontsize=10, color='blue')
ax5.set_title("Distance range = " + str(int(100*Dmins[4])) + " - " + str(int(100*Dmaxs[4])) + " cm", fontsize=10, color='blue')
ax6.set_title("Distance range = " + str(int(100*Dmins[5])) + " - " + str(int(100*Dmaxs[5])) + " cm", fontsize=10, color='blue')

ax5.set_xlabel(r"$\hat{v}_i(t) \cdot \hat{v}_j(t)$")
ax6.set_xlabel(r"$\hat{v}_i(t) \cdot \hat{v}_j(t)$")
ax1.set_ylabel("Probability density")
ax3.set_ylabel("Probability density")
ax5.set_ylabel("Probability density")

plt.savefig("C:/Users/Justin/Desktop/Fig-S8_alignment2.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after




