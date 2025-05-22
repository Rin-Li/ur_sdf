import torch
import trimesh
import numpy as np
import os
#UR 3e DH parameters
# A = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# alpha = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

A = [0.0, -0.24355, -0.2131, 0.0, 0.0, 0.0]
d = [0.15185, 0, 0, 0.13105, 0.08535, 0.0921]
alpha = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]
theta_min = [-6.283, -6.283, -6.283, -6.283, -6.283, -6.283]
theta_max = [6.283, 6.283, 6.283, 6.283, 6.283, 6.283]
# UR 3e visual offsets
offsets = {
        "base":      [0.0, 0.0, 0.0, 0.0, 0.0, np.pi],
        "shoulder":  [0.0, 0.0, 0.0, np.pi / 2, 0.0, np.pi],
        "upperarm":  [0.0, 0.0, 0.120, np.pi/2, 0.0, -3 * np.pi/2],
        "forearm":   [0.0, 0.0, 0.027, np.pi/2, 0.0, -np.pi/2],
        "wrist1":    [0.0, 0.0, -0.104, np.pi/2, 0.0, 0.0],
        "wrist2":    [0.0, 0.0, -0.08535, 0.0, 0.0, 0.0],
        "wrist3":    [0.0, 0.0, -0.0921, np.pi/2, 0.0, 0.0],
    }
# Link order for UR3e
link_order = ['base', 'shoulder', 'upperarm', 'forearm', 'wrist1', 'wrist2', 'wrist3']

class URRobot(torch.nn.Module):
    def __init__(self, device='cpu', mesh_path='ur3e/model/'):
        super().__init__()
        self.device = device
        self.mesh_path = mesh_path
        self.meshes = self.load_mesh()
        self.robot, self.robot_faces, self.robot_normals = zip(*[
            self.meshes[link] for link in link_order if link in self.meshes
        ])
    
    def visual_offset(self, name, batch_size=1):
        x, y, z, roll, pitch, yaw = offsets[name]
        dtype = torch.float32
        device = self.device

        Rx = torch.tensor([
            [1, 0, 0, 0],
            [0, np.cos(roll), -np.sin(roll), 0],
            [0, np.sin(roll),  np.cos(roll), 0],
            [0, 0, 0, 1]
        ], dtype=dtype, device=device)

        Ry = torch.tensor([
            [np.cos(pitch), 0, np.sin(pitch), 0],
            [0, 1, 0, 0],
            [-np.sin(pitch), 0, np.cos(pitch), 0],
            [0, 0, 0, 1]
        ], dtype=dtype, device=device)

        Rz = torch.tensor([
            [np.cos(yaw), -np.sin(yaw), 0, 0],
            [np.sin(yaw),  np.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=dtype, device=device)

        T = torch.tensor([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=dtype, device=device)

        single_T = T @ Rz @ Ry @ Rx
        # single_T = Rz @ Ry @ Rx
        return single_T.unsqueeze(0).repeat(batch_size, 1, 1)
        
    # Get the transformation matrix for each link vertices and normal
    def get_transformation_vertices_normals(self, vertices, normals, T, batch_size, name):
        vertices = vertices.repeat(batch_size, 1, 1)
        normals = normals.repeat(batch_size, 1, 1)
        T_offset = self.visual_offset(name, batch_size)
        # vertices = torch.matmul(torch.matmul(T, T_offset), vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        # normals = torch.matmul(torch.matmul(T, T_offset), normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        vertices = torch.matmul(T, vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        normals = torch.matmul(T, normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        return vertices, normals
    
    # Forward for UR
    def forward(self, pose, theta):
        
        batch_size = theta.shape[0]
        T = self.get_transformations_each_link(pose, theta)
        transformation_vertices, transformation_normals = zip(*[
            self.get_transformation_vertices_normals(self.robot[idx], self.robot_normals[idx], T[idx], batch_size, link_order[idx])
            for idx in range(len(self.robot))
        ])

        return transformation_vertices + transformation_normals
        
    # Transformations matrix
    def get_transformations_each_link(self,pose, theta):
        batch_size = theta.shape[0]
        T_prev_to_cur = []
        for idx in range(6):
            T_prev_to_cur.append(self.forward_kinematics(A[idx], torch.tensor(alpha[idx], dtype=torch.float32, device=self.device),
                                        d[idx], theta[:, idx], batch_size).float())        
        pose_to_each_link = [pose]
        for i in range(6):
            pose_to_each_link.append(torch.matmul(pose_to_each_link[i], T_prev_to_cur[i]))
        
        return pose_to_each_link
    
    def get_robot_mesh(self, vertices_list, face):
        robot_mesh = []
        for i in range(len(vertices_list)):
            mesh = trimesh.Trimesh(vertices=vertices_list[i], faces=face[i])
            robot_mesh.append(mesh)
        return robot_mesh
    
    # DH transformation matrix for each link
    def forward_kinematics(self, A, alpha, D, theta, batch_size=1):
        
        theta = theta.view(batch_size, -1)
        alpha = alpha*torch.ones_like(theta)
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_alpha = torch.cos(alpha)
        s_alpha = torch.sin(alpha)

        l_1_to_l = torch.cat([c_theta, -s_theta * c_alpha, s_theta * s_alpha, A * torch.ones_like(c_theta)* c_theta,
                             s_theta, c_theta * c_alpha, -c_theta * s_alpha, A * torch.ones_like(c_theta) * s_theta,
                             torch.zeros_like(s_theta), s_alpha, c_alpha, D * torch.ones_like(c_theta),
                              torch.zeros_like(s_theta), torch.zeros_like(s_theta), torch.zeros_like(s_theta), torch.ones_like(s_theta)], dim=1).reshape(batch_size, 4, 4)
        
        return l_1_to_l
    
    def get_forward_robot_mesh(self, pose, theta):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)
        
        vertices_list = [[
            outputs[0][i], outputs[1][i], outputs[2][i], outputs[3][i],
            outputs[4][i], outputs[5][i], outputs[6][i]
        ] for i in range(batch_size)]
        
        mesh = [self.get_robot_mesh(vertices, self.robot_faces) for vertices in vertices_list]
        return mesh
        
       
    
    # Load the mesh files
    def load_mesh(self):
        mesh_files = [f for f in os.listdir(self.mesh_path) if f.endswith('.stl')]
        meshes = {}

        for mesh_file in mesh_files:
            full_path = os.path.join(self.mesh_path, mesh_file)
            mesh = trimesh.load(full_path)
            name = os.path.splitext(mesh_file)[0]

            tmp = torch.ones(len(mesh.vertices), 1).float()
            vertices = torch.cat((torch.FloatTensor(mesh.vertices), tmp), dim=-1).to(self.device)
            normals = torch.cat((torch.FloatTensor(mesh.vertex_normals), tmp), dim=-1).to(self.device)
            faces = torch.LongTensor(mesh.faces).to(self.device)

            if name in offsets:
                T_offset = self.visual_offset(name, batch_size=1)[0]
                vertices = (T_offset @ vertices.T).T
                normals = (T_offset @ normals.T).T

            meshes[name] = [vertices, faces, normals]

        return meshes

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ur_robot = URRobot(device='cuda' if torch.cuda.is_available() else 'cpu')
    print("Loaded meshes:")
    # for name, mesh in meshes.items():
    #     print(f"Loaded mesh: {name}")
    #     print(f"Vertices: {mesh[0].shape}, Faces: {mesh[1].shape}, Normals: {mesh[2].shape}")
    
    theta = torch.zeros(1, 6).to(device)
    print(f"Random theta: {theta}")
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).expand(len(theta),-1,-1).float()
    print(f"Pose: {pose}")
    robot_mesh = ur_robot.get_forward_robot_mesh(pose, theta)
    robot_mesh = np.sum(robot_mesh)
    os.makedirs('output_meshes', exist_ok=True)
    trimesh.exchange.export.export_mesh(robot_mesh, os.path.join('output_meshes',f"whole_body_levelset_0.stl"))
    robot_mesh.show()


if __name__ == "__main__":
    main()
