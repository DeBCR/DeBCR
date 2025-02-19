import os
import matplotlib.pyplot as plt
import random

# visualization for two images
def subShow3(IMG1, IMG2, IMG3):
    
    color = 'inferno'
    
    plt.subplot(1,3,1)
    plt.imshow(IMG1, cmap=color)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(IMG2, cmap=color)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(IMG3, cmap=color)
    plt.axis('off')
    plt.show()

def subShow(IMG1, IMG2):
    
    color = 'inferno'
        
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(IMG1, cmap=color)
    plt.subplot(1,2,2)
    plt.imshow(IMG2, cmap=color)
    plt.show()
    plt.close()

def save_grid(pred_list, w_list, o_list, output_path, domain, eval_results, NUM=5, color='inferno'):
    num_images = NUM
    random_indices = random.sample(range(len(w_list)), num_images)
    
    # Format the title string using eval_results
    title_str = f'all test: PSNR: {eval_results[0]:.2f}, SSIM: {eval_results[1]:.4f}, RMSE: {eval_results[2]:.4f}'

    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 3, 9))
    
    # Set the figure title
    fig.suptitle(title_str, fontsize=16)

    for i, idx in enumerate(random_indices):
        axes[0, i].imshow(w_list[idx], cmap=color)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].axis('on')
            axes[0, i].set_ylabel('Input', fontsize=12, labelpad=10)
            axes[0, i].yaxis.set_ticks([])  
            axes[0, i].yaxis.set_ticklabels([])  
            axes[0, i].xaxis.set_ticks([])  
            axes[0, i].xaxis.set_ticklabels([]) 

    for i, idx in enumerate(random_indices):
        axes[1, i].imshow(pred_list[idx], cmap=color)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].axis('on')
            axes[1, i].set_ylabel('Prediction', fontsize=12, labelpad=10)
            axes[1, i].yaxis.set_ticks([])  
            axes[1, i].yaxis.set_ticklabels([])  
            axes[1, i].xaxis.set_ticks([])  
            axes[1, i].xaxis.set_ticklabels([]) 

    for i, idx in enumerate(random_indices):
        axes[2, i].imshow(o_list[idx], cmap=color)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].axis('on')
            axes[2, i].set_ylabel('Ground Truth', fontsize=12, labelpad=10)
            axes[2, i].yaxis.set_ticks([])  
            axes[2, i].yaxis.set_ticklabels([])  
            axes[2, i].xaxis.set_ticks([])  
            axes[2, i].xaxis.set_ticklabels([]) 
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to make room for the title
    svg_filename = os.path.join(output_path, f"{str(domain)}.png")
    plt.savefig(svg_filename, bbox_inches='tight', pad_inches=0, dpi=96)
    plt.close()
    
def show_grid(pred_list, w_list, o_list, NUM=5, color='inferno'):
    num_images = NUM
    random_indices = random.sample(range(len(w_list)), num_images)

    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 3, 9))
    
    for i, idx in enumerate(random_indices):
        axes[0, i].imshow(w_list[idx], cmap=color)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].axis('on')
            axes[0, i].set_ylabel('Input', fontsize=12, labelpad=10)
            axes[0, i].yaxis.set_ticks([])  
            axes[0, i].yaxis.set_ticklabels([])  
            axes[0, i].xaxis.set_ticks([])  
            axes[0, i].xaxis.set_ticklabels([]) 

    for i, idx in enumerate(random_indices):
        axes[1, i].imshow(pred_list[idx], cmap=color)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].axis('on')
            axes[1, i].set_ylabel('Prediction', fontsize=12, labelpad=10)
            axes[1, i].yaxis.set_ticks([])  
            axes[1, i].yaxis.set_ticklabels([])  
            axes[1, i].xaxis.set_ticks([])  
            axes[1, i].xaxis.set_ticklabels([]) 

    for i, idx in enumerate(random_indices):
        axes[2, i].imshow(o_list[idx], cmap=color)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].axis('on')
            axes[2, i].set_ylabel('Ground Truth', fontsize=12, labelpad=10)
            axes[2, i].yaxis.set_ticks([])  
            axes[2, i].yaxis.set_ticklabels([])  
            axes[2, i].xaxis.set_ticks([])  
            axes[2, i].xaxis.set_ticklabels([]) 
    
    plt.tight_layout()
    plt.show()