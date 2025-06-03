import trimesh
import glob
import os
import numpy as np
from mesh_to_sdf import mesh_to_voxels
import skimage


mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/model/*.stl"
mesh_files = glob.glob(mesh_path)
voxel_resolution = 128
for mf in mesh_files:
    mesh_name = mf.split('/')[-1].split('_')[0]
    print(mesh_name)
    scene = trimesh.Scene()
    mesh_origin = trimesh.load(mf)
    # mesh_origin.visual.face_colors = [255,0,0,150]
    center = mesh_origin.bounding_box.centroid
    scale = 2 / np.max(mesh_origin.bounding_box.extents)
    voxels = mesh_to_voxels(mesh_origin,
                            voxel_resolution=voxel_resolution,
                            surface_point_method='scan',
                            sign_method='normal',
                            scan_count=100,
                            scan_resolution=400,
                            sample_point_count=10000000,
                            normal_sample_count=100,
                            pad=True,
                            check_result=False)
    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0.0,spacing=(2/voxel_resolution,2/voxel_resolution,2/voxel_resolution))
    mesh_voxelized = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh_voxelized.visual.face_colors = [0,0,255,150]
    mesh_voxelized.vertices = mesh_voxelized.vertices/scale
    mesh_voxelized.vertices = mesh_voxelized.vertices - mesh_voxelized.bounding_box.centroid +center
    print(mesh_voxelized.vertices.shape)
    # scene.add_geometry(mesh_voxelized)
    # scene.show()

    base_dir = os.path.dirname(os.path.realpath(__file__))

    save_path = os.path.join(base_dir, 'voxel_128')


    os.makedirs(save_path, exist_ok=True)
    trimesh.exchange.export.export_mesh(mesh_voxelized, os.path.join(save_path,f'{mesh_name}.stl'))
