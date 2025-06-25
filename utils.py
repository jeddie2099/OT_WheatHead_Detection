
# This file contains some functions and other tools needed throughout the project 

import os
import shutil 
import imageio
import torch
import numpy as np

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
