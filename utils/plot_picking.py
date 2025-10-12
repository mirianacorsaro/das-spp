import numpy as np
import matplotlib.pyplot as plt

def plot_events(picking_list, image, cmap = None, vmin = None, vmax = None, fq = 200., gl = 5., time_points = 12000, num_channels = 4324):

    index_p = [(event[1], event[0]) for event in picking_list if event[2] == 1]
    index_p = list(zip(*index_p))

    index_s = [(event[1], event[0]) for event in picking_list if event[2] == 2]
    index_s = list(zip(*index_s))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.title('DAS data')
    dist =  np.arange(400) * 5 * 10 * 1e-3
    time =  np.arange(1200) / 200. * 10
    plt.imshow(image.cpu().squeeze(0).permute(1,2,0), extent =[time.min(), time.max(), dist.max(), dist.min()],  aspect="auto", interpolation="none")
    plt.ylabel("Distance (km)")
    plt.xlabel("Time (s)")

    plt.subplot(2, 1, 2)
    plt.title('Phases Picking')
    plt.imshow(image.cpu().squeeze(0).permute(1,2,0))
    if len(index_p) > 0:
        plt.scatter(index_p[0], index_p[1], c='b', s=1, marker='.') 
        plt.scatter([], [], c="b", label="P") 
    if len(index_s) > 0:
        plt.scatter(index_s[0], index_s[1], c='r', s=1, marker='.') 
        plt.scatter([], [], c="b", label="S") 
    plt.ylabel("Num. of Channels")
    plt.xlabel("Time")
    plt.show()
