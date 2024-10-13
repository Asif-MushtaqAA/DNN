import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Function to generate grid
def generate_grid(x_range, y_range, grid_size):
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    grid_x, grid_y = np.meshgrid(x, y)
    return grid_x, grid_y

def denormalize_inferred_field(inferred_field, global_min, global_max):
    denormalized_field = np.empty_like(inferred_field)
    
    for i in range(inferred_field.shape[2]):
        denormalized_field[:, :, i] = inferred_field[:, :, i] * (global_max[i] - global_min[i]) + global_min[i]
    
    denormalized_field[np.isnan(inferred_field)] = np.nan
    return denormalized_field

# Function to plot inferred variables
def plot_inferred_variables(grid_x, grid_y, grid_z, title):
    num_channels = grid_z.shape[2]
    fig, axes = plt.subplots(1, num_channels, figsize=(20, 5))
    
    for i in range(num_channels):
        cs = axes[i].contourf(grid_x, grid_y, grid_z[:, :, i], levels=200, cmap='turbo')
        axes[i].set_title(f'{title} - Channel {i + 1}')
        fig.colorbar(cs, ax=axes[i], orientation='vertical')
    
    plt.tight_layout()
    plt.show()
    
# Define region of interest
x_range = [-0.2, 1.5]
y_range = [-0.5, 0.5]
grid_size = 150

# Load global min and max values
global_min = np.load('./global_min.npy')
global_max = np.load('./global_max.npy')

# Generate grid
grid_x, grid_y = generate_grid(x_range, y_range, grid_size)

def main(airfoil_folder, mach_folder, angle_folder):
    # Load SDF (previously generated and saved)
    sdf_file = os.path.join('./sdf_data', f'{airfoil_folder}.npy')
    airfoil_contour = np.load(sdf_file)

    # load DNN inferred data
    inferred_file = os.path.join('./inferred_data', f'{airfoil_folder}_{mach_folder}_{angle_folder}.npy')
    inferred_z = np.load(inferred_file)
    inferred_z[airfoil_contour < 0] = np.nan
    denormalized_field = denormalize_inferred_field(inferred_z, global_min, global_max)

    # Plot flow variables before normalization
    plot_inferred_variables(grid_x, grid_y, inferred_z, f'DNN Inferred - {airfoil_folder}_{mach_folder}_{angle_folder}')
    plot_inferred_variables(grid_x, grid_y, denormalized_field, f'Denormalized Field - {airfoil_folder}_{mach_folder}_{angle_folder}')


    
# Example implementation in console
# Place sdf array and inferred data in the respective folders. Give DNN inferred field and denormalised flow field.

# from DNN_Visualiser import main
# main(58, 0.5, 4)