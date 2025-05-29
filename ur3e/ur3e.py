import torch
import trimesh
import os
import numpy as np

theta_min = [-6.283] * 6
theta_max = [ 6.283] * 6

link_order = ['base', 'shoulder', 'upperarm', 'forearm', 'wrist1', 'wrist2', 'wrist3']

visual_offset = {
            'base':     (0, 0, np.pi, 0, 0, 0),
            'shoulder': (0, 0, np.pi, 0, 0, 0),
            'upperarm': (np.pi/2, 0, -np.pi/2, 0, 0, 0.12),
            'forearm':  (np.pi/2, 0, -np.pi/2, 0, 0, 0.027),
            'wrist1':   (np.pi/2, 0, 0, 0, 0, -0.104),
            'wrist2':   (0, 0, 0, 0, 0, -0.08535),
            'wrist3':   (np.pi/2, 0, 0, 0, 0, -0.0921),
        }

kinematic = {
    'base'      : (0, 0, np.pi, 0, 0, 0),
    'shoulder'  : (0, 0, 0, 0, 0, 0.15185),
    'upperarm'  : (np.pi/2, 0, 0, 0, 0, 0),
    'forearm'   : (0, 0, 0, -0.24355, 0, 0),
    'wrist1'    : (0, 0, 0, -0.2132, 0, 0.13105),
    'wrist2'    : (np.pi/2, 0, 0, 0, -0.08535, 0),
    'wrist3'    : (np.pi/2, np.pi, np.pi, 0, 0.0921, 0),
}

class URRobot(torch.nn.Module):
    def __init__(self, device='cpu', mesh_path='ur3e/model/'):
        super().__init__()
        self.device    = device
        self.mesh_path = mesh_path
        self.meshes    = self.load_mesh()
        self.robot, self.robot_faces, self.robot_normals = zip(*[
            self.meshes[link] for link in link_order
        ])

    def Rx(self, r):
        c, s = np.cos(r), np.sin(r)
        return torch.tensor([[1, 0, 0, 0],
                             [0, c,-s, 0],
                             [0, s, c, 0],
                             [0, 0, 0, 1]], dtype=torch.float32, device=self.device)

    def Ry(self, p):
        c, s = np.cos(p), np.sin(p)
        return torch.tensor([[ c, 0, s, 0],
                             [ 0, 1, 0, 0],
                             [-s, 0, c, 0],
                             [ 0, 0, 0, 1]], dtype=torch.float32, device=self.device)

    def Rz(self, y):
        c, s = np.cos(y), np.sin(y)
        return torch.tensor([[c,-s, 0, 0],
                             [s, c, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=torch.float32, device=self.device)

    def T(self, x, y, z):
        return torch.tensor([[1, 0, 0, x],
                             [0, 1, 0, y],
                             [0, 0, 1, z],
                             [0, 0, 0, 1]], dtype=torch.float32, device=self.device)

    def T_origin(self, roll, pitch, yaw, x, y, z):
        return (self.T(x, y, z) @
                self.Rz(yaw) @ self.Ry(pitch) @ self.Rx(roll))


    def get_transformations_each_link(self, pose, theta):
        B = theta.size(0)
        Ts = [[] for _ in range(7)]
        for b in range(B):
            T_trans = [self.T_origin(*kinematic[link]) for link in link_order]
            T_trans[0] = pose[b] @ T_trans[0]
            # each link like T01 = T @ Rz(theta)
            for idx in range(1, 7):
                T_trans[idx] @= self.Rz(theta[b, idx-1])
            # Transfor matrxi like T2 = T1 @ T12
            T = [T_trans[0]]
            for idx in range(1, 7):
                T.append(T[idx - 1] @ T_trans[idx])
            for i, Ti in enumerate(T):
                Ts[i].append(Ti)
        return [torch.stack(t, 0) for t in Ts]

    def _transform_vn(self, v, n, T, B):
        v = v.repeat(B, 1, 1)
        n = n.repeat(B, 1, 1)
        v = (T @ v.transpose(2,1)).transpose(1,2)[:,:,:3]
        n = (T @ n.transpose(2,1)).transpose(1,2)[:,:,:3]
        return v, n

    def forward(self, pose, theta):
        B = theta.size(0)
        T_link = self.get_transformations_each_link(pose, theta)
        verts, norms = [], []
        for i, link in enumerate(link_order):
            T_vis = T_link[i] @ self.T_origin(*visual_offset[link])
            v, n  = self._transform_vn(self.robot[i], self.robot_normals[i],
                                       T_vis, B)
            verts.append(v); norms.append(n)
        return verts + norms


    def _make_mesh_batch(self, v_list, f_list):
        return [trimesh.Trimesh(vertices=v.cpu().numpy(),
                                faces=f.cpu().numpy(),
                                process=False)
                for v, f in zip(v_list, f_list)]

    def get_forward_robot_mesh(self, pose, theta):
        B = pose.size(0)
        out = self.forward(pose, theta)
        verts = out[:7]
        batches = []
        for b in range(B):
            meshes = self._make_mesh_batch([v[b] for v in verts],
                                           self.robot_faces)
            batches.append(trimesh.util.concatenate(meshes))
        return batches


    def load_mesh(self):
        meshes = {}
        for f in os.listdir(self.mesh_path):
            if not f.endswith('.stl'):
                continue
            m = trimesh.load(os.path.join(self.mesh_path, f))
            name = os.path.splitext(f)[0]
            ones = torch.ones(len(m.vertices), 1)
            v = torch.tensor(m.vertices, dtype=torch.float32)
            n = torch.tensor(m.vertex_normals, dtype=torch.float32)
            meshes[name] = [
                torch.cat((v, ones), -1).to(self.device),
                torch.tensor(m.faces, dtype=torch.long).to(self.device),
                torch.cat((n, ones), -1).to(self.device)
            ]
        return meshes



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ur = URRobot(device=device, mesh_path='ur3e/model/')

    theta = torch.ones(1, 6, device=device)     
    pose  = torch.eye(4, device=device).unsqueeze(0)

    mesh = ur.get_forward_robot_mesh(pose, theta)[0]
    os.makedirs('output_meshes', exist_ok=True)
    mesh.export('output_meshes/ur3e_zero_pose.stl')
    print('Export to output_meshes/ur3e_zero_pose.stl')


if __name__ == '__main__':
    main()
