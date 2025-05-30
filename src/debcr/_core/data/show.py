import random
import matplotlib.pyplot as plt

def show(data: list, slices=[-1], titles=[], figsize=(5,5), cmap='inferno', show_titles=True, show_ids=True, transpose=False):
    """
    Display selected slices from multiple 3D image volumes using matplotlib.

    This function visualizes specific z-slices from a list of 3D NumPy arrays (volumes),
    arranged in a grid. Each column represents one dataset; each row represents one slice.
    Useful for side-by-side comparisons of different volumes or models.

    Parameters
    ----------
    data : list of np.ndarray
        List of 3D arrays (e.g., shape (Z, X, Y)) to visualize.
    slices : list of int, optional
        Indices of z-slices to display. Use `-1` to select a random slice.
        The number of rows will be equal to the length of this list.
    titles : list of str, optional
        Titles for each dataset (i.e., per column). If fewer than `data`, blank titles are used.
    figsize : tuple of int, optional
        Size of each subplot in inches (width, height). Default is (5, 5).
    cmap : str, optional
        Colormap used to display the images. Default is 'inferno'.
    show_titles : bool, optional
        If True, displays the dataset title above each image. Default is True.
    show_ids : bool, optional
        If True, appends the slice index to the title (e.g., "[12]"). Default is True.
    transpose : bool, optional
        If True, transposes the grid layout so datasets are shown as rows instead of columns.

    Returns
    -------
    None
        Displays the image grid using `matplotlib.pyplot`.
    """
    
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