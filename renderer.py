import os
import glob
import multiprocessing
from pathlib import Path

import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from jaxtyping import Float


Mesh = trimesh.Trimesh

ModelNet40_LABELS = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]
ModelNet40_LABELS2INDEX = {label: i for i, label in enumerate(ModelNet40_LABELS)}


def normalize_mesh(mesh: Mesh) -> Mesh:
    """
    """
    mesh.apply_translation(-mesh.bounding_sphere.center)
    mesh.apply_scale(1 / mesh.bounding_sphere.primitive.radius)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 1]))
    return mesh


def plot_points(points: Float[torch.Tensor, "batch 3"], color: str = 'red') -> None:
    """
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        points[:, 0], 
        points[:, 1], 
        points[:, 2], color=color
    )
    plt.axis('equal') # preserve aspect ratio
    plt.show()


def generate_equiangular_rays(n: int) -> Float[torch.Tensor, "batch 3"]:
    """
    :param n: angular resolution
    """
    theta, phi = np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = np.sin(phi_grid) * np.cos(theta_grid)
    y = np.sin(phi_grid) * np.sin(theta_grid)
    z = np.cos(phi_grid)
    directions = np.stack((x, y, z), axis=-1)
    return directions


def render_distances(mesh: Mesh, directions: Float[np.ndarray, "batch 3"]) -> Float[torch.Tensor, "n n"]:
    """
    :param n: angular resolution
    """
    n, m, _ = directions.shape
    directions = directions.reshape(-1, 3)
    origins = np.array([mesh.bounding_sphere.center] * len(directions))

    locations, index_ray, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
    
    distances = np.zeros(len(directions))
    for loc, idx in zip(locations, index_ray):
        distance = np.linalg.norm(loc - mesh.bounding_sphere.center)
        distances[idx] = max(distances[idx], distance)
    distances = distances.reshape(n, m)
    return distances


def render_modelnet40_distances(filename: Path | str, output: Path | str):
    """
    """
    print('Rendering ', filename)
    mesh = trimesh.load_mesh(filename)
    mesh = normalize_mesh(mesh)
    directions = generate_equiangular_rays(64)
    distances = render_distances(mesh, directions)
    print('Done rendering ', filename)
    np.save(f'{output}/{Path(filename.name).stem}.npy', distances)
    return distances


def render_images(mesh: Mesh, ncount=64):
    """
    Render images at random angles distance 2 from the center of the mesh
    """
    def random_point_on_sphere(radius):
        phi, theta = -np.random.uniform(-np.pi/4, np.pi/4), np.random.uniform(0, 2 * np.pi)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.array([x, y, z])

    scene = mesh.scene()
    images = []
    for i in range(ncount):
        camera_position = random_point_on_sphere(2.5)
        camera_transform = trimesh.geometry.align_vectors([0, 0, 1], camera_position)
        camera_transform[:3, 3] = camera_position
        scene.graph[scene.camera.name] = camera_transform
        image = scene.save_image(resolution=(128, 128), visible=True)
        image = Image.open(BytesIO(image))
        images.append(image)
    return images
    

def render_modelnet40_images(filename: Path | str, output: Path | str):
    """
    """
    print('Rendering ', filename)
    mesh = trimesh.load_mesh(filename)
    mesh = normalize_mesh(mesh)
    images = render_images(mesh)
    print('Done rendering ', filename)
    for i, image in enumerate(images):
        image.save(f'{output}/{Path(filename.name).stem}_{i:04d}.png')
    return images


def render_modelnet40(render_func: callable, path: Path | str, output: Path| str, partition='train'):
    """
    """
    os.makedirs(output, exist_ok=True)
    filenames = []
    for label in ['car']:#odelNet40_LABELS:
        filenames_label = list(glob.glob(f'{path}/{label}/{partition}/*.off', recursive=True))
        filenames_label = sorted(filenames_label)[:300]
        for filename in filenames_label:
            filenames.append((Path(filename), Path(output)))
    filenames = sorted(filenames)

    print(f'Found {len(filenames)} files')
    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(render_func, filenames)
    print('Done rendering')


if __name__ == "__main__":
    #cat = 'airplane'
    #mesh = trimesh.load_mesh(f'/home/gtangg12/data/ModelNet40_aligned/{cat}/train/{cat}_0050.off')
    #mesh = normalize_mesh(mesh)
    #viewer = mesh.show()
    '''
    directions = generate_equiangular_rays(32)
    import matplotlib.pyplot as plt

    distances_ref = render_distances(mesh, directions)
    distances = np.load('/home/gtangg12/data/scnn/airplane_0001.npy')
    assert np.allclose(distances, distances_ref)
    
    from mayavi import mlab
    mlab.mesh(
        directions[:, :, 0], 
        directions[:, :, 1], 
        directions[:, :, 2], scalars=distances, colormap='coolwarm'
    )
    mlab.show()
    '''
    #images = render_images(mesh, ncount=1)
    #print(images[0].size)
    #images[0].show()
    #exit()
    #render_modelnet40(render_modelnet40_distances, '/home/gtangg12/data/ModelNet40_aligned/', '/home/gtangg12/data/scnn/s2', partition='train')
    render_modelnet40(render_modelnet40_images, '/home/gtangg12/data/ModelNet40_aligned', '/home/gtangg12/data/scnn/images_test', partition='test')