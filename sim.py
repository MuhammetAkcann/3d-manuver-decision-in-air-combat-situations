import glob
import os
import re

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np

import sys

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


for infile in sorted(glob.glob('*.txt'), key=numericalSort):
    print("Current File Being Processed is: " + infile)

### TO DO
## Senkronizasyondaki sıkıntıyı gider


global counter, np_rewards
counter = 0

folder = f"{sys.argv[1]}/"
print(sys.argv)
print(sys.argv[1])


def update_lines(num, dataLines, lines):
    global counter, np_rewards
    # print(zip(lines, dataLines))
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])

    string = "OR: " + str(np_rewards[counter % np_rewards.shape[0]][0])[0:5]
    string2 = "DR: " + str(np_rewards[counter % np_rewards.shape[0]][1])[0:5]

    print(string)
    print(string2)

    counter += 1
    return lines


sims = sorted(os.listdir(folder), key=numericalSort, reverse=True)
print(sims)

for i in sims:
    if "sim" in i:
        continue
    nparr = np.load(folder + "locations_" + i.split("_")[1].split(".")[0] + ".npy")
    np_rewards = np.load(folder + "rewards_" + i.split("_")[1].split(".")[0] + ".npy")
    print(max(np.sum(np_rewards, axis=1) / 2))
    print(i)
    if max(np.sum(np_rewards, axis=1) / 2) < 8:
        pass
        #continue
    input()
    print(i)
    counter = 0
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_xlim3d([-40, 100.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-40.0, 100.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-40.0, 100.0])
    ax.set_zlabel('Z')
    blue = np.array([nparr[0][0], nparr[1][0], nparr[2][0]])
    red = np.array([nparr[0][1], nparr[1][1], nparr[2][1]])
    data = [blue, red]
    print(len(data[0][0]))
    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
    line_ani = animation.FuncAnimation(fig, update_lines, len(data[0][0]), fargs=(data, lines),
                                       interval=50, blit=True)

    plt.show()
    plt.close()
