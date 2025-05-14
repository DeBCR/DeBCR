import random
import matplotlib.pyplot as plt

def show_slices(data_list: list, slice_id=-1, titles=[], figsize=(5,5), cmap='inferno', show_titles=True, show_ids=True):
    
    if slice_id == -1:
        slice_id = random.randint(0, data_list[0].shape[0]-1)
    
    n = len(data_list)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0]*n, figsize[1]))
    for idx in range(n):
        axes[idx].imshow(data_list[idx][slice_id], cmap=cmap)    
        axes[idx].axis('off')

    if show_titles:
        for idx in range(n):
            title = f'{titles[idx]} ' if idx < len(titles) else f''
            title += f' [{slice_id}]' if show_ids else '' 
            axes[idx].title.set_text(title)
    
    plt.show()