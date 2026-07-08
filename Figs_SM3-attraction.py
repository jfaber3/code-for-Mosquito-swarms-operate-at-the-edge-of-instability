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




#FINDING ACCELERATION MAGNITUDE VS DISTANCE FROM WEIGHTED MEAN
def find_acc_mag(pos_data, acc_data, data_mask, clr):
    print("Finding Magnitudes")
    test_pwrs = np.array( [0, 1, 2, 3] )
    Dist1, Dist2, Dist3, Dist4 = [], [], [], []
    A = []
    T = pos_data.shape[1]
    for t in range(T):
        mask_t = data_mask[:, t]
        Pos = pos_data[mask_t, t]
        Acc = acc_data[mask_t, t]
        N = len(Pos)
        if N <= 1:
            continue
        D_vecs = Pos[None, :, :] - Pos[:, None, :]
        D_mag = np.linalg.norm(D_vecs, axis=2)
        np.fill_diagonal(D_mag, np.inf)
        
        #calculating weighting mean fields
        weights = np.where( D_mag[None, :, :] == np.inf, 0, D_mag[None, :, :] ** (-test_pwrs[:, None, None]) )
        wmf_vecs = np.sum(D_vecs[None, :, :, :] * weights[:, :, :, None], axis=2) / np.sum( weights[:, :, :, None], axis=2)
        wmf_norm = np.linalg.norm(wmf_vecs, axis=2)

        A.extend(np.linalg.norm(Acc, axis=1))
        Dist1.extend( wmf_norm[0, :] * 100 )
        Dist2.extend( wmf_norm[1, :] * 100 )
        Dist3.extend( wmf_norm[2, :] * 100 )
        Dist4.extend( wmf_norm[3, :] * 100 )
    Num_Bins = 15  #15
    Max_Dist = 30  #30 cm
    stat_type = 'mean'
    acc_data1,  bin_edges, _  = binned_statistic(np.array(Dist1), np.array(A), statistic=stat_type, bins=Num_Bins, range=(0, Max_Dist))
    acc_data2,  bin_edges, _  = binned_statistic(np.array(Dist2), np.array(A), statistic=stat_type, bins=Num_Bins, range=(0, Max_Dist))
    acc_data3,  bin_edges, _  = binned_statistic(np.array(Dist3), np.array(A), statistic=stat_type, bins=Num_Bins, range=(0, Max_Dist))
    acc_data4,  bin_edges, _  = binned_statistic(np.array(Dist4), np.array(A), statistic=stat_type, bins=Num_Bins, range=(0, Max_Dist))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax1.plot(bin_centers, acc_data1, "o-",  color=clr, markersize=3)
    ax2.plot(bin_centers, acc_data2, "o-",  color=clr, markersize=3)
    ax3.plot(bin_centers, acc_data3, "o-",  color=clr, markersize=3)
    ax4.plot(bin_centers, acc_data4, "o-",  color=clr, markersize=3)




clrs =[ "#000000", "#ff0000", "#d000ff", "#2b00ff", "#00b3ff", "#00bb5a", "#ffb700" ]


fig  = plt.figure(figsize=(5, 5), constrained_layout=True)
ax1  = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax2  = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax3  = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax4  = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)

find_acc_mag(pos_data1, acc_data1, data_mask1, clrs[1])
find_acc_mag(pos_data2, acc_data2, data_mask2, clrs[2])
find_acc_mag(pos_data3, acc_data3, data_mask3, clrs[3])
find_acc_mag(pos_data4, acc_data4, data_mask4, clrs[4])
find_acc_mag(pos_data5, acc_data5, data_mask5, clrs[5])
find_acc_mag(pos_data6, acc_data6, data_mask6, clrs[6])

ax1.set_title("$\gamma = 0$")
ax2.set_title("$\gamma = 1$")
ax3.set_title("$\gamma = 2$")
ax4.set_title("$\gamma = 3$")
ax1.set_xlabel(r"$\langle |\vec{D}_i| \rangle \ (cm)$")
ax2.set_xlabel(r"$\langle |\vec{D}_i| \rangle \ (cm)$")
ax3.set_xlabel(r"$\langle |\vec{D}_i| \rangle \ (cm)$")
ax4.set_xlabel(r"$\langle |\vec{D}_i| \rangle \ (cm)$")
ax1.set_ylabel(r"$\langle |\frac{d\vec{v}_i}{dt}| \rangle \ (m/s^2)$")
ax2.set_ylabel(r"$\langle |\frac{d\vec{v}_i}{dt}| \rangle \ (m/s^2)$")
ax3.set_ylabel(r"$\langle |\frac{d\vec{v}_i}{dt}| \rangle \ (m/s^2)$")
ax4.set_ylabel(r"$\langle |\frac{d\vec{v}_i}{dt}| \rangle \ (m/s^2)$")

ax1.set_xlim(0, 30)
ax2.set_xlim(0, 30)
ax3.set_xlim(0, 30)
ax4.set_xlim(0, 30)
ax1.set_ylim(1, 4)
ax2.set_ylim(1, 4)
ax3.set_ylim(1, 4)
ax4.set_ylim(1, 4)

plt.savefig("C:/Users/Justin/Desktop/Fig-S5_attraction.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after






