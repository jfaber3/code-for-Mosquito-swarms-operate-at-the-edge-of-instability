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



def plot_everything(num_bins, vel_data1, vel_data2, vel_data3, vel_data4, vel_data5, vel_data6):
    vx1, vy1, vz1, mag_v1 = flatten_data(vel_data1, data_mask1)
    vx2, vy2, vz2, mag_v2 = flatten_data(vel_data2, data_mask2)
    vx3, vy3, vz3, mag_v3 = flatten_data(vel_data3, data_mask3)
    vx4, vy4, vz4, mag_v4 = flatten_data(vel_data4, data_mask4)
    vx5, vy5, vz5, mag_v5 = flatten_data(vel_data5, data_mask5)
    vx6, vy6, vz6, mag_v6 = flatten_data(vel_data6, data_mask6)

    all_x = np.concatenate((vx1, vx2, vx3, vx4, vx5, vx6))
    all_y = np.concatenate((vy1, vy2, vy3, vy4, vy5, vy6))
    all_z = np.concatenate((vz1, vz2, vz3, vz4, vz5, vz6))
    all_mag = np.concatenate((mag_v1, mag_v2, mag_v3, mag_v4, mag_v5, mag_v6))

    ax1.hist(vx1, bins=num_bins, color=clrs[1], histtype='step', density=True)
    ax1.hist(vx2, bins=num_bins, color=clrs[2], histtype='step', density=True)
    ax1.hist(vx3, bins=num_bins, color=clrs[3], histtype='step', density=True)
    ax1.hist(vx4, bins=num_bins, color=clrs[4], histtype='step', density=True)
    ax1.hist(vx5, bins=num_bins, color=clrs[5], histtype='step', density=True)
    ax1.hist(vx6, bins=num_bins, color=clrs[6], histtype='step', density=True)

    ax2.hist(vy1, bins=num_bins, color=clrs[1], histtype='step', density=True)
    ax2.hist(vy2, bins=num_bins, color=clrs[2], histtype='step', density=True)
    ax2.hist(vy3, bins=num_bins, color=clrs[3], histtype='step', density=True)
    ax2.hist(vy4, bins=num_bins, color=clrs[4], histtype='step', density=True)
    ax2.hist(vy5, bins=num_bins, color=clrs[5], histtype='step', density=True)
    ax2.hist(vy6, bins=num_bins, color=clrs[6], histtype='step', density=True)

    ax3.hist(vz1, bins=num_bins, color=clrs[1], histtype='step', density=True)
    ax3.hist(vz2, bins=num_bins, color=clrs[2], histtype='step', density=True)
    ax3.hist(vz3, bins=num_bins, color=clrs[3], histtype='step', density=True)
    ax3.hist(vz4, bins=num_bins, color=clrs[4], histtype='step', density=True)
    ax3.hist(vz5, bins=num_bins, color=clrs[5], histtype='step', density=True)
    ax3.hist(vz6, bins=num_bins, color=clrs[6], histtype='step', density=True)

    ax4.hist(mag_v1, bins=num_bins, color=clrs[1], histtype='step', density=True)
    ax4.hist(mag_v2, bins=num_bins, color=clrs[2], histtype='step', density=True)
    ax4.hist(mag_v3, bins=num_bins, color=clrs[3], histtype='step', density=True)
    ax4.hist(mag_v4, bins=num_bins, color=clrs[4], histtype='step', density=True)
    ax4.hist(mag_v5, bins=num_bins, color=clrs[5], histtype='step', density=True)
    ax4.hist(mag_v6, bins=num_bins, color=clrs[6], histtype='step', density=True)

    ax1.set_ylabel("Probability density")
    ax2.set_ylabel("Probability density")
    ax3.set_ylabel("Probability density")
    ax4.set_ylabel("Probability density")

    return all_x, all_y, all_z, all_mag






clrs =[ "#000000", "#ff0000", "#d000ff", "#2b00ff", "#00b3ff", "#00bb5a", "#ffb700", ]
num_bins_r = 50  #position
num_bins_v = 50  #velocity
num_bins_a = 50  #acceleration




#POSITION FIGURE
fig = plt.figure(figsize=(5, 5), constrained_layout=True)
ax1  = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax2  = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax3  = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax4  = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)

rx_all, ry_all, rz_all, r_mag_all = plot_everything(num_bins_r, pos_data1, pos_data2, pos_data3, pos_data4, pos_data5, pos_data6)

r_limit = 0.5
ax1.set_xlim(-r_limit, r_limit)
ax2.set_xlim(-r_limit, r_limit)
ax3.set_xlim(0, 1.5)
ax4.set_xlim(0, 1.5)
ax1.set_xlabel("$r_x \ (m)$")
ax2.set_xlabel("$r_y \ (m)$")
ax3.set_xlabel("$r_z \ (m)$")
ax4.set_xlabel(r"$|\vec{r}| \ (m)$")

plt.savefig("C:/Users/Justin/Desktop/Fig-S1_position_stats.pdf", dpi=300)






#VELOCITY FIGURE
fig = plt.figure(figsize=(5, 5), constrained_layout=True)
ax1  = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax2  = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax3  = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax4  = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)

vx_all, vy_all, vz_all, v_mag_all = plot_everything(num_bins_v, vel_data1, vel_data2, vel_data3, vel_data4, vel_data5, vel_data6)

v_limit = 1.2
ax1.set_xlim(-v_limit, v_limit)
ax2.set_xlim(-v_limit, v_limit)
ax3.set_xlim(-v_limit, v_limit)
ax4.set_xlim(0, 2)
ax1.set_xlabel("$v_x \ (m/s)$")
ax2.set_xlabel("$v_y \ (m/s)$")
ax3.set_xlabel("$v_z \ (m/s)$")
ax4.set_xlabel(r"$|\vec{v}| \ (m/s)$")

#adding x, y to the z plot for comparison
ax3.hist(vx_all, bins=num_bins_v, color="black", histtype='step', density=True, zorder=0, ls='dotted')
ax3.hist(vy_all, bins=num_bins_v, color="red", histtype='step', density=True, zorder=0, ls='dotted')

#all_speed = 0.50 +- 0.14 m/s
plt.savefig("C:/Users/Justin/Desktop/Fig-S2_velocity_stats.pdf", dpi=300)





#ACCELERATION FIGURE
fig = plt.figure(figsize=(5, 5), constrained_layout=True)
ax1  = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax2  = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax3  = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax4  = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)

ax_all, ay_all, az_all, a_mag_all = plot_everything(num_bins_a, acc_data1, acc_data2, acc_data3, acc_data4, acc_data5, acc_data6)

a_limit = 8
ax1.set_xlim(-a_limit, a_limit)
ax2.set_xlim(-a_limit, a_limit)
ax3.set_xlim(-a_limit, a_limit)
ax4.set_xlim(0, 10)
ax1.set_xlabel("$a_x \ (m/s^2)$")
ax2.set_xlabel("$a_y \ (m/s^2)$")
ax3.set_xlabel("$a_z \ (m/s^2)$")
ax4.set_xlabel(r"$|\vec{a}| \ (m/s^2)$")

#adding x, y to the z plot for comparison
ax3.hist(ax_all, bins=num_bins_a, color="black", histtype='step', density=True, zorder=0, ls='dotted')
ax3.hist(ay_all, bins=num_bins_a, color="red", histtype='step', density=True, zorder=0, ls='dotted')

plt.savefig("C:/Users/Justin/Desktop/Fig-S3_acceleration_stats.pdf", dpi=300)



plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after






