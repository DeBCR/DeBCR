import random
import matplotlib.pyplot as plt

def show(data: list, slices=[-1], titles=[], figsize=(5,5), cmap='inferno', show_titles=True, show_ids=True, transpose=False):

    n = len(data)
    ns = len(slices)
    
    for i in range(ns):
        slices[i] = slices[i] if slices[i] != -1 else random.randint(0, data[0].shape[0]-1)

    nx, ny = ns, n
    if transpose:
        nx, ny = ny, nx
    fig, axes = plt.subplots(nx, ny, figsize=(figsize[0]*ny, figsize[1]*nx))
    
    axes = [axes] if nx*ny==1 else axes.flatten()
    
    for i, axis in enumerate(axes):
        ix, iy = i // ny, i % ny
        if transpose:
            ix, iy = iy, ix
        axis.imshow(data[iy][slices[ix]], cmap=cmap)
        axis.axis('off')

    if show_titles:
        for i, axis in enumerate(axes):
            ix, iy = i // ny, i % ny
            if transpose:
                ix, iy = iy, ix
            title = f'{titles[iy]} ' if iy < len(titles) else f''
            title += f' [{slices[ix]}]' if show_ids else ''
            axis.title.set_text(title)
    
    plt.show()