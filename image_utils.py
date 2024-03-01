
import numpy as np
import matplotlib.pyplot as plt



def show_image(image: np.ndarray, title:str='', block=True) -> None:
    
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    ax = plt.imshow(image, cmap='grey', vmin=0, vmax=255, interpolation='none')
    fig.axes[0].grid(False)
    plt.show(block=block)

def show_images(images: list[np.ndarray], titles:list[str], suptitle:str='') -> None:
    fig, axes = plt.subplots(1,len(images))
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='grey', vmin=0, vmax=255, interpolation='none')
        axes[i].grid(False)
        axes[i].set_title(titles[i])
        
    fig.suptitle(suptitle, fontsize=16)


