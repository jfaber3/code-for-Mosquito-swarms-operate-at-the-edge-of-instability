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




def find_avg_dot_product(pos_data, acc_data, data_mask, NN_count, gamma=0):
    print("calculating dot products")
    T = pos_data.shape[1]
    all_dots = [[] for _ in NN_count]
    visual_dots, visual_axis_dots = [], []

    for t in range(T):
        mask_t = data_mask[:, t]
        Pos = pos_data[mask_t, t]
        Acc = acc_data[mask_t, t]
        N = len(Pos)
        if N <= 1:
            continue

        Pos[:, 2], Acc[:, 2] = 0, 0    #setting all z components to 0

        D_vecs = Pos[None, :, :] - Pos[:, None, :]    # (N, N, 3)
        D_mag  = np.linalg.norm(D_vecs, axis=2)       # (N, N)
        np.fill_diagonal(D_mag, np.inf)

        #from data directly
        acc_norm = np.linalg.norm(Acc, axis=1)
        acc_norm[acc_norm == 0] = np.inf
        acc_unit = Acc / acc_norm[:, None]            # (N, 3)

        # Visual cue dot products to origin
        vis_norm = np.linalg.norm(Pos, axis=1)
        vis_norm[vis_norm == 0] = np.inf
        visual_unit = -Pos / vis_norm[:, None]
        visual_dots.extend(np.sum(visual_unit * acc_unit, axis=1).tolist())

        # Visual cue dot product of acc ONTO z axis. throwing out z component
        Pos_xy = np.copy(Pos)
        Pos_xy[:, 2] = 0
        vis_norm2 = np.linalg.norm(Pos_xy, axis=1)
        vis_norm2[vis_norm2 == 0] = np.inf
        visual_unit2 = -Pos_xy / vis_norm2[:, None]
        visual_axis_dots.extend(np.sum(visual_unit2 * acc_unit, axis=1).tolist())

        # k-nearest neighbor dot products with gamma weighting
        sorted_idx = np.argsort(D_mag, axis=1)        # (N, N)
        sorted_mag = D_mag[np.arange(N)[:, None], sorted_idx]  # (N, N) sorted distances

        for ki, k in enumerate(NN_count):
            kNN      = sorted_idx[:, :k]                        # (N, k)
            kD_mag   = sorted_mag[:, :k]                        # (N, k) sorted distances
            weights  = kD_mag ** (-gamma)                       # (N, k)
            NN_vecs  = Pos[kNN] - Pos[:, None, :]               # (N, k, 3)
            wmf_vecs = (weights[:, :, None] * NN_vecs).sum(axis=1)  # (N, 3)
            wmf_norm = np.linalg.norm(wmf_vecs, axis=1)
            wmf_norm[wmf_norm == 0] = np.inf
            wmf_unit = wmf_vecs / wmf_norm[:, None]
            dots = np.sum(wmf_unit * acc_unit, axis=1)  # (N,)
            all_dots[ki].extend(dots.tolist())
    return np.median(visual_dots), np.median(visual_axis_dots), np.array([np.median(d) for d in all_dots])




NN_count = np.linspace(1, 25, 13, dtype=int)   #np.linspace(1, 15, 15, dtype=int)

DP_vis1, DP_vis1_b, DP1_gamma0  = find_avg_dot_product(pos_data1, acc_data1, data_mask1, NN_count, gamma=0)
DP_vis2, DP_vis2_b, DP2_gamma0  = find_avg_dot_product(pos_data2, acc_data2, data_mask2, NN_count, gamma=0)
DP_vis3, DP_vis3_b, DP3_gamma0  = find_avg_dot_product(pos_data3, acc_data3, data_mask3, NN_count, gamma=0)
DP_vis4, DP_vis4_b, DP4_gamma0  = find_avg_dot_product(pos_data4, acc_data4, data_mask4, NN_count, gamma=0)
DP_vis5, DP_vis5_b, DP5_gamma0  = find_avg_dot_product(pos_data5, acc_data5, data_mask5, NN_count, gamma=0)
DP_vis6, DP_vis6_b, DP6_gamma0  = find_avg_dot_product(pos_data6, acc_data6, data_mask6, NN_count, gamma=0)

_, _, DP1_gamma1  = find_avg_dot_product(pos_data1, acc_data1, data_mask1, NN_count, gamma=1)
_, _, DP2_gamma1  = find_avg_dot_product(pos_data2, acc_data2, data_mask2, NN_count, gamma=1)
_, _, DP3_gamma1  = find_avg_dot_product(pos_data3, acc_data3, data_mask3, NN_count, gamma=1)
_, _, DP4_gamma1  = find_avg_dot_product(pos_data4, acc_data4, data_mask4, NN_count, gamma=1)
_, _, DP5_gamma1  = find_avg_dot_product(pos_data5, acc_data5, data_mask5, NN_count, gamma=1)
_, _, DP6_gamma1  = find_avg_dot_product(pos_data6, acc_data6, data_mask6, NN_count, gamma=1)

_, _, DP1_gamma2  = find_avg_dot_product(pos_data1, acc_data1, data_mask1, NN_count, gamma=2)
_, _, DP2_gamma2  = find_avg_dot_product(pos_data2, acc_data2, data_mask2, NN_count, gamma=2)
_, _, DP3_gamma2  = find_avg_dot_product(pos_data3, acc_data3, data_mask3, NN_count, gamma=2)
_, _, DP4_gamma2  = find_avg_dot_product(pos_data4, acc_data4, data_mask4, NN_count, gamma=2)
_, _, DP5_gamma2  = find_avg_dot_product(pos_data5, acc_data5, data_mask5, NN_count, gamma=2)
_, _, DP6_gamma2  = find_avg_dot_product(pos_data6, acc_data6, data_mask6, NN_count, gamma=2)

_, _, DP1_gamma3  = find_avg_dot_product(pos_data1, acc_data1, data_mask1, NN_count, gamma=3)
_, _, DP2_gamma3  = find_avg_dot_product(pos_data2, acc_data2, data_mask2, NN_count, gamma=3)
_, _, DP3_gamma3  = find_avg_dot_product(pos_data3, acc_data3, data_mask3, NN_count, gamma=3)
_, _, DP4_gamma3  = find_avg_dot_product(pos_data4, acc_data4, data_mask4, NN_count, gamma=3)
_, _, DP5_gamma3  = find_avg_dot_product(pos_data5, acc_data5, data_mask5, NN_count, gamma=3)
_, _, DP6_gamma3  = find_avg_dot_product(pos_data6, acc_data6, data_mask6, NN_count, gamma=3)


mrkr_size = 4
clrs =[ "#000000", "#ff0000", "#d000ff", "#2b00ff", "#00b3ff", "#00bb5a", "#ffb700" ]

#POSITION FIGURE
fig = plt.figure(figsize=(5, 5), constrained_layout=True)
ax1  = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax2  = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax3  = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax4  = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)





ax1.axhline(y=DP_vis1_b, ls='dotted', color=clrs[1])
ax1.axhline(y=DP_vis2_b, ls='dotted', color=clrs[2])
ax1.axhline(y=DP_vis3_b, ls='dotted', color=clrs[3])
ax1.axhline(y=DP_vis4_b, ls='dotted', color=clrs[4])
ax1.axhline(y=DP_vis5_b, ls='dotted', color=clrs[5])
ax1.axhline(y=DP_vis6_b, ls='dotted', color=clrs[6])

ax2.axhline(y=DP_vis1_b, ls='dotted', color=clrs[1])
ax2.axhline(y=DP_vis2_b, ls='dotted', color=clrs[2])
ax2.axhline(y=DP_vis3_b, ls='dotted', color=clrs[3])
ax2.axhline(y=DP_vis4_b, ls='dotted', color=clrs[4])
ax2.axhline(y=DP_vis5_b, ls='dotted', color=clrs[5])
ax2.axhline(y=DP_vis6_b, ls='dotted', color=clrs[6])

ax3.axhline(y=DP_vis1_b, ls='dotted', color=clrs[1])
ax3.axhline(y=DP_vis2_b, ls='dotted', color=clrs[2])
ax3.axhline(y=DP_vis3_b, ls='dotted', color=clrs[3])
ax3.axhline(y=DP_vis4_b, ls='dotted', color=clrs[4])
ax3.axhline(y=DP_vis5_b, ls='dotted', color=clrs[5])
ax3.axhline(y=DP_vis6_b, ls='dotted', color=clrs[6])

ax4.axhline(y=DP_vis1_b, ls='dotted', color=clrs[1])
ax4.axhline(y=DP_vis2_b, ls='dotted', color=clrs[2])
ax4.axhline(y=DP_vis3_b, ls='dotted', color=clrs[3])
ax4.axhline(y=DP_vis4_b, ls='dotted', color=clrs[4])
ax4.axhline(y=DP_vis5_b, ls='dotted', color=clrs[5])
ax4.axhline(y=DP_vis6_b, ls='dotted', color=clrs[6])



ax1.plot(NN_count, DP1_gamma0, "o-", markersize=mrkr_size, color=clrs[1])
ax1.plot(NN_count, DP2_gamma0, "o-", markersize=mrkr_size, color=clrs[2])
ax1.plot(NN_count, DP3_gamma0, "o-", markersize=mrkr_size, color=clrs[3])
ax1.plot(NN_count, DP4_gamma0, "o-", markersize=mrkr_size, color=clrs[4])
ax1.plot(NN_count, DP5_gamma0, "o-", markersize=mrkr_size, color=clrs[5])
ax1.plot(NN_count, DP6_gamma0, "o-", markersize=mrkr_size, color=clrs[6])

ax2.plot(NN_count, DP1_gamma1, "o-", markersize=mrkr_size, color=clrs[1])
ax2.plot(NN_count, DP2_gamma1, "o-", markersize=mrkr_size, color=clrs[2])
ax2.plot(NN_count, DP3_gamma1, "o-", markersize=mrkr_size, color=clrs[3])
ax2.plot(NN_count, DP4_gamma1, "o-", markersize=mrkr_size, color=clrs[4])
ax2.plot(NN_count, DP5_gamma1, "o-", markersize=mrkr_size, color=clrs[5])
ax2.plot(NN_count, DP6_gamma1, "o-", markersize=mrkr_size, color=clrs[6])

ax3.plot(NN_count, DP1_gamma2, "o-", markersize=mrkr_size, color=clrs[1])
ax3.plot(NN_count, DP2_gamma2, "o-", markersize=mrkr_size, color=clrs[2])
ax3.plot(NN_count, DP3_gamma2, "o-", markersize=mrkr_size, color=clrs[3])
ax3.plot(NN_count, DP4_gamma2, "o-", markersize=mrkr_size, color=clrs[4])
ax3.plot(NN_count, DP5_gamma2, "o-", markersize=mrkr_size, color=clrs[5])
ax3.plot(NN_count, DP6_gamma2, "o-", markersize=mrkr_size, color=clrs[6])

ax4.plot(NN_count, DP1_gamma3, "o-", markersize=mrkr_size, color=clrs[1])
ax4.plot(NN_count, DP2_gamma3, "o-", markersize=mrkr_size, color=clrs[2])
ax4.plot(NN_count, DP3_gamma3, "o-", markersize=mrkr_size, color=clrs[3])
ax4.plot(NN_count, DP4_gamma3, "o-", markersize=mrkr_size, color=clrs[4])
ax4.plot(NN_count, DP5_gamma3, "o-", markersize=mrkr_size, color=clrs[5])
ax4.plot(NN_count, DP6_gamma3, "o-", markersize=mrkr_size, color=clrs[6])

ax1.set_title("$\gamma = 0$")
ax2.set_title("$\gamma = 1$")
ax3.set_title("$\gamma = 2$")
ax4.set_title("$\gamma = 3$")

ax1.set_xlim(0, max(NN_count)+1)
ax2.set_xlim(0, max(NN_count)+1)
ax3.set_xlim(0, max(NN_count)+1)
ax4.set_xlim(0, max(NN_count)+1)
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)
ax3.set_ylim(0, 1)
ax4.set_ylim(0, 1)

ax1.set_xlabel("No. neighbors")
ax2.set_xlabel("No. neighbors")
ax3.set_xlabel("No. neighbors")
ax4.set_xlabel("No. neighbors")
#ax1.set_ylabel("Mean dot product")
#ax2.set_ylabel("Mean dot product")
#ax3.set_ylabel("Mean dot product")
#ax4.set_ylabel("Mean dot product")
ax1.set_ylabel("Median dot product")
ax2.set_ylabel("Median dot product")
ax3.set_ylabel("Median dot product")
ax4.set_ylabel("Median dot product")

plt.savefig("C:/Users/Justin/Desktop/Fig-S4_neighbor_count_TEST.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after






