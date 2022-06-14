import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import random

from config import *
env_init()

from framework.utils import *
from framework.Environment import Environment
from framework.GRASS import SOURCE_NAME

from Region import Region


DEBUG = False

T = 600
bound = Bound(57992, 54747, -14955, -11471)
source = (56978.3098189104,-12406.60548812005)
environment = Environment(bound, 'fuel', 'samplefm100', 'evi', 'samplevs', 'sampleth', 'dem', source)
true_p = environment.generate_wildfire(T)
plt.imshow(true_p)

n_sensors = 400

np.random.seed(3)
row_idx = np.random.choice(environment.rows, n_sensors)
col_idx = np.random.choice(environment.cols, n_sensors)
node_indexes = [[row_idx[i], col_idx[i]] for i in range(n_sensors)]


region = Region(node_indexes, environment)
plt.figure(figsize=(10,10))
plt.imshow(region.sub_regions)
plt.scatter(col_idx, row_idx, c='r', marker='D', s=10)
plt.show()


radius = 20

on_fire = []
pre = []

points = []
step  = 15
for i in range(0, T, step):
    on_fire.append(environment.get_on_fire(i))
    fire_zone = np.where(on_fire[-1]> 0)
    fire_zone = set(zip(fire_zone[0], fire_zone[1]))
    print(f'On-fire area: {len(fire_zone)}')


    action = []
    # action = np.random.choice([True, False], n_sensors, p=[0.05, 1-0.05])

    for j, p in enumerate(node_indexes):
        neighbor = set([(p[0] + a, p[1] + b) for a in range(-radius, radius + 1) for b in range(-radius, radius + 1)])
        if neighbor.intersection(fire_zone) and np.random.rand()<0.7:
            action.append(True)
        elif np.random.rand()<0.3:
            action.append(True)
        else:
            action.append(False)


    points.append([[col, row] for col, row, a in zip(col_idx, row_idx, action) if a])

    if i == 0:
        region.model_update(action, i , SOURCE_NAME)
        predict, ros = region.predict(SOURCE_NAME, i, step, 'predict')
    else:
        region.model_update(action, i , 'predict')
        predict, ros = region.predict('predict', i , step, 'predict')
    pre.append(predict)
    print(f'Predict area: {(predict>0).sum()}')

a = np.array(on_fire)
b = np.array(pre)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

def animate(i):
    fig.suptitle(f'$\Delta t={step}, T = {i * step}$', fontsize=16)
    ax[0].clear()
    ax[1].clear()
    ax[0].imshow(a[i])
    ax[1].imshow(b[i])
    ax[1].scatter([p[0] for p in points[i]], [p[1] for p in points[i]])
    e = np.sum(np.absolute(a[i] - b[i]))
    ax[0].set_title('Propogation')
    ax[1].set_title('Prediction, e=%d, $e/\sqrt{A}=$%.2f' % (e, e / np.sqrt(np.sum(a[i])+1)))
    plt.tight_layout()

writer = animation.writers['ffmpeg']
writer = writer(fps=2, metadata=dict(artist='Me'), bitrate=900)
ani = FuncAnimation(fig, animate, len(a), interval=1000)
ani.save(f"results/spread_ani_step_{step}.mp4", writer=writer)

# sa = np.sum(a, axis=0)
# sb = np.sum(b, axis=0)
# fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
# fig.suptitle(f'$\Delta t={step}$', fontsize=16)
# ax[0].imshow(np.where(sa>0, sa+10, 0), cmap='hot')
# ax[1].imshow(np.where(sb>0, sb+10, 0), cmap='hot')
# ax[0].set_title('Propogation')
# ax[1].set_title('Prediction')
# plt.tight_layout()
# fig.savefig(f"results/spread_step_{step}.jpg", dpi=fig.dpi)
#
# e = np.array([np.sum(np.absolute(a[i] - b[i])) for i in range(len(a))])
# A = np.array([np.sum(an) for an in a])
# fig, ax = plt.subplots(1,3, sharex=True)
# ax[0].plot(e)
# ax[0].set_title('Error')
# ax[1].plot(e/(A+1))
# ax[1].set_title('Error/Area')
# ax[2].plot(e/np.sqrt(A+1))
# ax[2].set_title('Error/sqrt(Area)')
# fig.savefig(f"results/err_step_{step}.jpg", dpi=fig.dpi)

# fire_zone = np.where(true_p> 0)
# fire_zone = set(zip(fire_zone[0], fire_zone[1]))
# region_size = len(fire_zone)
# true_fire = environment.get_on_fire(T)
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
#
#
#
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
#         pre, ros = simulation.app.fusion_center.field_reconstruct(SOURCE_NAME, 0, T, 'predict',  f'_{n_sensors}')
#         pre_size = np.where(pre + environment.source  > 0, 1, 0)
#         axs[n, i].imshow(pre)
#         axs[n, i].scatter([n.index.col for n in random_select],
#                           [n.index.row for n in random_select], color='yellow')
#         axs[n, i].set_title(f'{k} sensors, {q}, error={np.sum(np.absolute(pre_size - true_fire))}')
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

