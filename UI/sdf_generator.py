import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
import time

#1.SDF Generator
def load_airfoil_geometry(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    coordinates = []
    for line in lines:
        data = line.split()
        if len(data) == 2:
            x = float(data[0])
            y = float(data[1])
            coordinates.append([x, y])
    return np.array(coordinates)

def create_sdf_image(airfoil_coords, x_range, y_range, grid_size):
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    airfoil_polygon = Polygon(airfoil_coords)
    tree = cKDTree(airfoil_coords)
    distances, _ = tree.query(grid_points)
    inside_mask = np.array([airfoil_polygon.contains(Point(p)) for p in grid_points])
    sdf_values = np.where(inside_mask, -distances, distances)
    sdf_image = sdf_values.reshape(grid_x.shape)
    
    return grid_x, grid_y, sdf_image

def generate_sdf(airfoil_number, sdf_output_folder = 'E:/SU/Dis/CFD/sdf_images', data_geometry_folder = 'E:/SU/Dis/libairfoil-master/data_geometry'):
    
    start_m = time.time()
    
    x_range = [-0.2, 1.5]
    y_range = [-0.5, 0.5]
    grid_size = 150

    if not os.path.exists(sdf_output_folder):
        os.makedirs(sdf_output_folder)

    airfoil_path = os.path.join(data_geometry_folder, f'{airfoil_number}.txt')
    airfoil_coords = load_airfoil_geometry(airfoil_path)
    grid_x, grid_y, sdf_image = create_sdf_image(airfoil_coords, x_range, y_range, grid_size)
    
    sdf_output_path = os.path.join(sdf_output_folder, f'{airfoil_number}.npy')
    np.save(sdf_output_path, sdf_image)
    
    end_m = time.time()
    elapsed_total = end_m - start_m
    print(f'Time taken for sdf generation: {elapsed_total:.2f} seconds')
    
    # Plot and save the SDF image for verification with a readable background
    plt.figure(figsize=(10, 5), facecolor='white')  # Set the background color to white
    contour_levels = np.linspace(sdf_image.min(), sdf_image.max(), 20)
    contour = plt.contourf(grid_x, grid_y, sdf_image, levels=contour_levels, cmap='RdYlBu')
    plt.colorbar(contour, label='Signed Distance to Airfoil Surface')
    plt.plot(airfoil_coords[:, 0], airfoil_coords[:, 1], 'k', lw=2, label='Airfoil Geometry')  # Plot airfoil geometry
    plt.title(f'Signed Distance Function (SDF) for Airfoil {airfoil_number}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(False)
    plt.savefig(os.path.join(sdf_output_folder, f'{airfoil_number}.png'), facecolor='white')
    plt.close()

    print(f'SDF for airfoil {airfoil_number} generated and saved.')
    
    return airfoil_coords, sdf_image, grid_x, grid_y, x_range, y_range

#Example Implementation in console
# Define appropriate paths for folder to store sdf arrays and location to aerofoil geometry in the generate_sdf function. "def generate_sdf(airfoil_number, sdf_output_folder = 'E:/SU/Dis/CFD/sdf_images', data_geometry_folder = 'E:/SU/Dis/libairfoil-master/data_geometry'):"

#from sdf_generator import generate_sdf
#generate_sdf(58)
# returns 6 variables as well!