import numpy as np
import code
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic

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

def flatten_data(data, data_mask):
    data_x = data[:, :, 0].flatten()[data_mask.flatten()]
    data_y = data[:, :, 1].flatten()[data_mask.flatten()]
    data_z = data[:, :, 2].flatten()[data_mask.flatten()]
    mag_data = np.linalg.norm(data, axis=2).flatten()[data_mask.flatten()]
    return data_x, data_y, data_z, mag_data




window_length = 15   #use 15  (min 5)
filt_order    = 3    #use 3   (min 2, max = window-1)

pos_data1, vel_data1, acc_data1, pos_CM1, vel_CM1, acc_CM1, data_mask1 = get_all_data(Data1, window_length, polyorder=filt_order)
pos_data2, vel_data2, acc_data2, pos_CM2, vel_CM2, acc_CM2, data_mask2 = get_all_data(Data2, window_length, polyorder=filt_order)
pos_data3, vel_data3, acc_data3, pos_CM3, vel_CM3, acc_CM3, data_mask3 = get_all_data(Data3, window_length, polyorder=filt_order)
pos_data4, vel_data4, acc_data4, pos_CM4, vel_CM4, acc_CM4, data_mask4 = get_all_data(Data4, window_length, polyorder=filt_order)
pos_data5, vel_data5, acc_data5, pos_CM5, vel_CM5, acc_CM5, data_mask5 = get_all_data(Data5, window_length, polyorder=filt_order)
pos_data6, vel_data6, acc_data6, pos_CM6, vel_CM6, acc_CM6, data_mask6 = get_all_data(Data6, window_length, polyorder=filt_order)


def find_velocity_correlations(pos_data, vel_data, acc_data, data_mask):
    print("Finding Alignment")
    T = pos_data.shape[1]
    Dist, Dot_prod = [], []
    for t in range(T):
        mask_t = data_mask[:, t]
        Pos = pos_data[mask_t, t]
        Vel = vel_data[mask_t, t]
        Acc = acc_data[mask_t, t]
        N = len(Pos)
        if N <= 1:
            continue
        D_vecs = Pos[None, :, :] - Pos[:, None, :]
        D_mag = np.linalg.norm(D_vecs, axis=2)
        Vel_unit = Vel/np.linalg.norm(Vel, axis=1)[:, None]
        Vel_Corr_Matrix = np.sum(Vel_unit[None, :, :] * Vel_unit[:, None, :], axis=2)
        D_mag_up_tri    =           D_mag[np.triu_indices_from(D_mag, k=0)]
        vel_corr_up_tri = Vel_Corr_Matrix[np.triu_indices_from(Vel_Corr_Matrix, k=0)]
        Dist.extend(D_mag_up_tri)
        Dot_prod.extend(vel_corr_up_tri)
    return np.array(Dist), np.array(Dot_prod)


Dist1, Corr1 = find_velocity_correlations(pos_data1, vel_data1, acc_data1, data_mask1)
Dist2, Corr2 = find_velocity_correlations(pos_data2, vel_data2, acc_data2, data_mask2)
Dist3, Corr3 = find_velocity_correlations(pos_data3, vel_data3, acc_data3, data_mask3)
Dist4, Corr4 = find_velocity_correlations(pos_data4, vel_data4, acc_data4, data_mask4)
Dist5, Corr5 = find_velocity_correlations(pos_data5, vel_data5, acc_data5, data_mask5)
Dist6, Corr6 = find_velocity_correlations(pos_data6, vel_data6, acc_data6, data_mask6)


clrs =[ "#000000", "#ff0000", "#d000ff", "#2b00ff", "#00b3ff", "#00bb5a", "#ffb700" ]
stat_type = 'mean'
Max_Dist = 0.1
Num_Bins = 15

R1,  bin_edges, _  = binned_statistic(Dist1, Corr1, statistic=stat_type, bins=Num_Bins, range=(0, Max_Dist))
R2,  bin_edges, _  = binned_statistic(Dist2, Corr2, statistic=stat_type, bins=Num_Bins, range=(0, Max_Dist))
R3,  bin_edges, _  = binned_statistic(Dist3, Corr3, statistic=stat_type, bins=Num_Bins, range=(0, Max_Dist))
R4,  bin_edges, _  = binned_statistic(Dist4, Corr4, statistic=stat_type, bins=Num_Bins, range=(0, Max_Dist))
R5,  bin_edges, _  = binned_statistic(Dist5, Corr5, statistic=stat_type, bins=Num_Bins, range=(0, Max_Dist))
R6,  bin_edges, _  = binned_statistic(Dist6, Corr6, statistic=stat_type, bins=Num_Bins, range=(0, Max_Dist))
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) * 100  #centimeters

fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True) 
ax.axhline(y=0, color='black', ls="dashed")
ax.plot(bin_centers, R1, "o-", color=clrs[1])
ax.plot(bin_centers, R2, "o-", color=clrs[2])
ax.plot(bin_centers, R3, "o-", color=clrs[3])
ax.plot(bin_centers, R4, "o-", color=clrs[4])
ax.plot(bin_centers, R5, "o-", color=clrs[5])
ax.plot(bin_centers, R6, "o-", color=clrs[6])
ax.set_xlabel("Distance (cm)")
ax.set_ylabel(r"$\langle \hat{v}_i(t) \cdot \hat{v}_j(t) \rangle$")


plt.savefig("C:/Users/Justin/Desktop/Fig-S7_alignment1.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after





