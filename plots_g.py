import numpy as np
import matplotlib.pyplot as plt

data = np.load("swarm_data_g.npz")
ss_array   = data["ss_array"]
g_values = data["g_values"].tolist()
l_values   = data["l_values"].tolist()

FONT_SIZE_LABEL  = 13
FONT_SIZE_TICK   = 8
FONT_SIZE_TITLE  = 15
Z_LIM            = (0, 50)
VIEW_ELEV        = 5
VIEW_AZIM        = 130
CMAP             = "viridis"
FIG_SIZE         = (8, 6)
# ─────────────────────────────────────────────────────────────────────────────

L_grid, n_grid = np.meshgrid(l_values, g_values)
 
fig = plt.figure(figsize=FIG_SIZE)
ax  = fig.add_subplot(projection="3d")
 
ax.plot_surface(L_grid, n_grid, ss_array, cmap=CMAP)
 
ax.set_xlabel("L",              fontsize=FONT_SIZE_LABEL)
ax.set_ylabel(r'$\gamma$',              fontsize=FONT_SIZE_LABEL)
ax.set_zlabel("Mean Swarm Size", fontsize=FONT_SIZE_LABEL)
ax.tick_params(labelsize=FONT_SIZE_TICK)
ax.set_zlim(*Z_LIM)
ax.set_ylim(min(g_values), max(g_values))
ax.view_init(VIEW_ELEV, VIEW_AZIM, 0)
 
plt.tight_layout()
plt.savefig("", dpi=150)
plt.savefig("/Users/annabelleboots/gamma_surface.pdf", dpi=300)

plt.show()
print("Saved swarm_figure.png")

