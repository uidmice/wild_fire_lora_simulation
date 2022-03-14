import matplotlib.pyplot as plt
import numpy as np
import random

from config import *
env_init()

from framework.utils import *
from framework.Environment import Environment
from Simulation import Simulation

DEBUG = False

T = 500
bound = Bound(57992, 54747, -14955, -11471)
source = (56978.3098189104,-12406.60548812005)
environment = Environment(bound, 'fuel', 'samplefm100', 'evi', 'samplevs', 'sampleth', 'dem', source)
true_p = environment.generate_wildfire(T)
plt.imshow(true_p)

n_sensors = 400

row_idx = np.random.choice(environment.rows, n_sensors)
col_idx = np.random.choice(environment.cols, n_sensors)
node_indexes = [Index(row_idx[i], col_idx[i]) for i in range(n_sensors)]
gateway_indexes = [Index(environment.rows//2, environment.cols//2)]

step_time = 6000  # ms
offset = 3000

simulation = Simulation(node_indexes, gateway_indexes, step_time, environment, offset=offset)
plt.scatter([n.index.col for n in simulation.nodes], [n.index.row for n in simulation.nodes])
plt.show()

num_steps = T * GRASS_TO_SIMPY_TIME_FACTOR / step_time

for i in range(3):
    send_index, received = simulation.step(random_policy(0.5, simulation))

# fire_zone = np.where(true_p> 0)
# fire_zone = set(zip(fire_zone[0], fire_zone[1]))
#
# extra = set()
# radius = 2
# for i in range(environment.rows):
#     for j in range(environment.cols):
#         if (i, j) not in fire_zone:
#             neighbor = set([(i+a, j + b) for a in range(-radius, radius+1) for b in range(-radius, radius+1)])
#             if neighbor.intersection(fire_zone):
#                 extra.update((i, j))
# fire_zone.update(extra)
#
# sensor_in = []
# sensor_out = []
# for s in simulation.nodes:
#     idx = (s.index.row, s.index.col)
#     if idx in fire_zone:
#         sensor_in.append(s)
#     else:
#         sensor_out.append(s)
# print(len(sensor_out))
# print(len(sensor_in))
#
# ratio = [0.2, 0.5, 0.8]
#
# max_k = int(min(len(sensor_in)/0.8, len(sensor_out)/0.8))



# repeat = 3
#
# K = [max_k//10, max_k//5, max_k//2, max_k]
# print(K)
# #
# fig, axs = plt.subplots(len(K), len(ratio), figsize=(15,15), sharex=True, sharey=True)
#
# for n, k in enumerate(K):
#     random_select_in = random.sample(sensor_in, int(k * max(ratio)))
#     random_select_out = random.sample(sensor_out, int(k * (1-min(ratio))))
#     for i, q in enumerate(sorted(ratio, reverse=True)):
#         simulation.reset()
#         n_in = int(k * q)
#         random_select_in = random.sample(random_select_in, n_in)
#         random_select = random.sample(random_select_out, k-n_in-1) + random_select_in
#         print(len(random_select))
#         for node in random_select:
#             data = environment.sense(node.index.row, node.index.col, 20)
#             data['x'] = node.location.x
#             data['y'] = node.location.y
#             info = PacketInformation(0, node.id, data, 20)
#             simulation.app.fusion_center.update(info)
#         pre, ros = simulation.app.fusion_center.field_reconstruct(T, f'_{n_sensors}')
#
#         axs[n, i].imshow(pre)
#         axs[n, i].scatter([n.index.col for n in random_select],
#                           [n.index.row for n in random_select], color='yellow')
#         axs[n, i].set_title(f'{k} sensors, {q}, error={np.sum(np.absolute(true_p - pre))/len(fire_zone):.2f}')
# plt.tight_layout()
# plt.show()
# plt.ion()
# fig = plt.figure(figsize=(10,10))
#
# ax = fig.add_subplot(1,1,1)
# im = ax.imshow(environment.T, alpha=.5, interpolation='bicubic', cmap='RdYlGn_r', origin='lower'
# ,extent=[-MAX_DISTANCE, MAX_DISTANCE, - MAX_DISTANCE, MAX_DISTANCE]
# )
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im.set_clim(353, 293)
# plt.colorbar(im, orientation='horizontal', pad=0.08)
# plt.axis('off')
# plt.draw()
# plt.show()
#
# deploy_ax = fig.add_axes(ax.get_position(), frameon=False)
# deploy_ax.set_xlim([-MAX_DISTANCE, MAX_DISTANCE])
# deploy_ax.set_ylim([-MAX_DISTANCE, MAX_DISTANCE])
# deploy_ax.scatter(0,0, s = 80, marker='X', c='r')
# for n in node_locations:
#     deploy_ax.scatter(n.x, n.y, s=20, c='blue')
#
# contour_ax = fig.add_axes(ax.get_position(), frameon=False)
# contour_ax.set_xlim([-MAX_DISTANCE, MAX_DISTANCE])
# contour_ax.set_ylim([-MAX_DISTANCE, MAX_DISTANCE])
# contour_ax.axes.xaxis.set_visible(False)
# contour_ax.axes.yaxis.set_visible(False)
#
# # plt.colorbar(im, orientation="vertical", pad=0.2)
#
# ax_text = fig.add_axes([0.1, 0.93, 0.1, 0.05])
# ax_text.axis("off")
# time_label = ax_text.text(0.5, 0.5, "0s", ha="left", va="top")
#
# writer = animation.writers['ffmpeg']
# writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=900)
# ani = animation.FuncAnimation(fig, environment.draw,400, fargs=(im, contour_ax, time_label, UPDATA_RATE),
#                                       interval=200)
# ani.save("demo.mp4", writer=writer)

# Z, tr = pickle.load(open("result/config2_update_18000_field_random_0.1.pkl", 'rb'))
# diff = Z - tr
# plt.ion()
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(1,1,1)
#
# for i in range(Z.shape[0]):
#     plt.imshow(diff[i], cmap='hot')
#     plt.colorbar()
#     plt.draw()
#     plt.pause(0.02)

