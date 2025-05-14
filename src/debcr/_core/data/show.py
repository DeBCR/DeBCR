import random
import matplotlib.pyplot as plt

def show(data: list, slices=[-1], titles=[], figsize=(5,5), cmap='inferno', show_titles=True, show_ids=True):

    ns = len(slices)
    for idx in range(ns):
        slices[idx] = slices[idx] if slices[idx] != -1 else random.randint(0, data[0].shape[0]-1)
       
    n = len(data)
    fig, axes = plt.subplots(ns, n, figsize=(figsize[0]*n, figsize[1]*ns))
    axes = axes.flatten()
    
    for idx, axis in enumerate(axes):
        slice_idx = idx // n
        data_idx  = idx % n
        axis.imshow(data[data_idx][slice_idx], cmap=cmap)
        axis.axis('off')

    if show_titles:
        for idx, axis in enumerate(axes):
            slice_idx = idx // n
            data_idx  = idx % n
            title = f'{titles[data_idx]} ' if data_idx < len(titles) else f''
            title += f' [{slices[slice_idx]}]' if show_ids else '' 
            axis.title.set_text(title)
    
    plt.show()