import torch
import trimesh
import numpy as np
import os
#UR 3e DH parameters
A = [0.0, -0.24355, -0.2131, 0.0, 0.0, 0.0]
d = [0.15185, 0, 0, 0.13105, 0.08535, 0.0921]
alpha = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]

class URRobot(torch.nn.Module):
    def __init__(self, device='cpu', mesh_path='ur3e/model/'):
        super().__init__()
        self.device = device
        self.mesh_path = mesh_path
        self.meshes = self.load_mesh()
        
        self.base = self.meshes['base'][0]
        self.base_normals = self.meshes['base'][2]
        
        self.shoulder = self.meshes['shoulder'][0]
        self.shoulder_normals = self.meshes['shoulder'][2]
        
        self.forearm = self.meshes['forearm'][0]
        self.forearm_normals = self.meshes['forearm'][2]
        
        self.upperarm = self.meshes['upperarm'][0]
        self.upperarm_normals = self.meshes['upperarm'][2]
        
        self.wrist1 = self.meshes['wrist1'][0]
        self.wrist1_normals = self.meshes['wrist1'][2]
        self.wrist2 = self.meshes['wrist2'][0]
        self.wrist2_normals = self.meshes['wrist2'][2]
        self.wrist3 = self.meshes['wrist3'][0]
        self.wrist3_normals = self.meshes['wrist3'][2]
    
    def forward(self, pose, theta):
        batch_size = theta.shape[0]
        base_vertices = self.base.repeat(batch_size, 1, 1)
        base_vertices = torch.matmul(pose,
                                      base_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        base_normals = self.base_normals.repeat(batch_size, 1, 1)
        base_normals = torch.matmul(pose,
                                      base_normals.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        
        
        
        
        
    
    # Transformations matrix
    def get_transformations_each_link(self,pose, theta):
        batch_size = theta.shape[0]
        T01 = self.forward_kinematics(A[0], torch.tensor(alpha[0], dtype=torch.float32, device=self.device),
                                        d[0], theta[:, 0], batch_size).float()

        T12 = self.forward_kinematics(A[1], torch.tensor(alpha[1], dtype=torch.float32, device=self.device),
                                        d[1], theta[:, 1], batch_size).float()
        T23 = self.forward_kinematics(A[2], torch.tensor(alpha[2], dtype=torch.float32, device=self.device),
                                        d[2], theta[:, 2], batch_size).float()
        T34 = self.forward_kinematics(A[3], torch.tensor(alpha[3], dtype=torch.float32, device=self.device),
                                        d[3], theta[:, 3], batch_size).float()
        T45 = self.forward_kinematics(A[4], torch.tensor(alpha[4], dtype=torch.float32, device=self.device),
                                        d[4], theta[:, 4], batch_size).float()
        T56 = self.forward_kinematics(A[5], torch.tensor(alpha[5], dtype=torch.float32, device=self.device),
                                        d[5], theta[:, 5], batch_size).float()

        pose_to_Tw0 = pose
        pose_to_T01 = torch.matmul(pose_to_Tw0, T01)
        pose_to_T12 = torch.matmul(pose_to_T01, T12)
        pose_to_T23 = torch.matmul(pose_to_T12, T23)
        pose_to_T34 = torch.matmul(pose_to_T23, T34)
        pose_to_T45 = torch.matmul(pose_to_T34, T45)
        pose_to_T56 = torch.matmul(pose_to_T45, T56)

        return [pose_to_Tw0,pose_to_T01,pose_to_T12,pose_to_T23,pose_to_T34,pose_to_T45,pose_to_T56]
    
    # DH transformation matrix for each link
    def forward_kinematics(self, A, alpha, D, theta, batch_size=1):
        theta = theta.view(batch_size, -1)
        alpha = alpha*torch.ones_like(theta)
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_alpha = torch.cos(alpha)
        s_alpha = torch.sin(alpha)

        l_1_to_l = torch.cat([c_theta, -s_theta, torch.zeros_like(s_theta), A * torch.ones_like(c_theta),
                                s_theta * c_alpha, c_theta * c_alpha, -s_alpha, -s_alpha * D,
                                s_theta * s_alpha, c_theta * s_alpha, c_alpha, c_alpha * D,
                                torch.zeros_like(s_theta), torch.zeros_like(s_theta), torch.zeros_like(s_theta), torch.ones_like(s_theta)], dim=1).reshape(batch_size, 4, 4)
        
        return l_1_to_l
    
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

            meshes[name] = [vertices, faces, normals]

        return meshes

def main():
    ur_robot = URRobot(device='cuda' if torch.cuda.is_available() else 'cpu')
    meshes = ur_robot.load_mesh()
    print("Loaded meshes:")
    # for name, mesh in meshes.items():
    #     print(f"Loaded mesh: {name}")
    #     print(f"Vertices: {mesh[0].shape}, Faces: {mesh[1].shape}, Normals: {mesh[2].shape}")

if __name__ == "__main__":
    main()
