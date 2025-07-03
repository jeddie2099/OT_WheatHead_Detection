
# This file contains some functions and other tools needed throughout the project 

import os
import shutil 
import imageio
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import math
import random

# Function for copying files from one folder to another
def copy_files(source_folder, file_list, destination_folder):
    for file_name in file_list:
        # Full path for the file in source folder 
        source_file_path = os.path.join(source_folder, file_name)
        
        # Full path for the file in destination folder
        destination_file_path = os.path.join(destination_folder, file_name)
        
        # Copy the file and make sure it was copied
        shutil.copy(source_file_path, destination_file_path)
        print(f'{file_name} successfully copied to {destination_folder}')


# Load images and normalize them to [0,1]
def load_image(fname):
    img = imageio.v2.imread(fname)  # RGB
    return img / 255.0  


# Transform an image to an RGB cloud, i.e. a sampled probability measure
def RGB_cloud(fname, sampling, dtype=torch.FloatTensor):
    A = load_image(fname)
    A = A[::sampling, ::sampling, :]
    return torch.from_numpy(A).type(dtype).view(-1, 3)

# Display the RGB cloud above as a 3D plot
def display_cloud(ax, x):
    x_ = x.detach().cpu().numpy()
    ax.scatter(x_[:, 0], x_[:, 1], x_[:, 2], s=25 * 500 / len(x_), c=x_)

# Display the image, using the RGB cloud as reference
def display_image(ax, x):
    W = int(np.sqrt(len(x)))
    x_ = x.view(W, W, 3).detach().cpu().numpy()
    ax.imshow(x_)


# Function to perform the color transfer between measures X_i and Y_j, as the gradient of some loss function 
def color_transfer(X_i, Y_j, loss, lr=1):
    """Flows along the gradient of the loss function.

    Parameters:
        loss ((x_i,y_j) -> torch float number):
            Real-valued loss function.
        lr (float, default = 1):
            Learning rate, i.e. time step.
    """

    # Parameters for the gradient descent
    Nsteps = 11

    # Make sure that we won't modify the reference samples
    x_i, y_j = X_i.clone(), Y_j.clone()

    # We're going to perform gradient descent on Loss(α, β)
    # wrt. the positions x_i of the diracs masses that make up α:
    x_i.requires_grad = True

    for i in range(Nsteps):  # Euler scheme ===============
        # Compute cost and gradient
        L_αβ = loss(x_i, y_j)
        [g] = torch.autograd.grad(L_αβ, [x_i])

        # in-place modification of the tensor's values
        x_i.data -= lr * len(x_i) * g
    return x_i

# Function that creates a panel of images belonging to different domains of the train, valid or test sets
def show_domain_images(csv_path, image_folder, n_domains=5, n_per_domain=2, random_state=42):
    #Read the given csv file containing the image and domain names
    df = pd.read_csv(csv_path)

    # Obtain unique domains from the selected 
    unique_domains = df['domain'].unique()

    if n_domains > len(unique_domains):
        raise ValueError(f"There are only {len(unique_domains)} domains available.")

    # Select 'n_domains' random domains from the ones available
    selected_domains = random.sample(list(unique_domains), n_domains)

    # Filter the dataframe for the selected domains
    filtered_df = df[df['domain'].isin(selected_domains)]

    # Take 'n_per_domain' images from every selected domain
    sample = (
        filtered_df.groupby("domain")
        .apply(lambda x: x.sample(n=min(n_per_domain, len(x)), random_state=random_state))
        .reset_index(drop=True)
    )

    # Adjusts the number of rows for the panel with 4-image rows
    total_images = len(sample)
    rows = math.ceil(total_images / 4)
    columns = min(total_images, 4)

    # Create the image panel with all the selected images, along with the name of the domain they belong to 
    fig, axs = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axs = axs.flatten() if total_images > 1 else [axs]

    for i, row in enumerate(sample.itertuples()):
        full_path = os.path.join(image_folder, row.image_name)
        try:
            img = Image.open(full_path)
            axs[i].imshow(img)
            axs[i].set_title(str(row.domain))
            axs[i].axis('off')
        except Exception as e:
            print(f"The image {full_path} couldn't be loaded: {e}")
            axs[i].axis('off')

    # Hide empty panels if there are less images 
    for j in range(len(sample), len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    return fig