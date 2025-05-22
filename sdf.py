# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# A minimal script to train a Bernstein-Polynomial SDF for a single STL object
# -----------------------------------------------------------------------------
import os, argparse, glob, numpy as np, trimesh, torch, skimage
import mesh_to_sdf                      # 依赖 https://github.com/marian42/mesh_to_sdf
from pathlib import Path

np.set_printoptions(threshold=np.inf)

CUR_DIR = Path(__file__).resolve().parent

# -------------------------- Bernstein-Polynomial SDF -------------------------
class BPSDF:
    def __init__(self, n_func, domain_min, domain_max, device):
        self.n_func, self.domain_min, self.domain_max, self.device = n_func, domain_min, domain_max, device
        self.model_path = CUR_DIR / "models"
        self.model_path.mkdir(exist_ok=True)

    # ---------- 1D Bernstein basis（与原实现完全一致，省略注释） ----------
    @staticmethod
    def _binom(n, k): return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))
    def _bernstein_1d(self, t, deriv=False):
        t = torch.clamp(t, 1e-4, 1 - 1e-4)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        comb = self._binom(torch.tensor(n, device=self.device), i)
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i
        if not deriv: return phi, None
        dphi = -comb * (n - i) * (1 - t).unsqueeze(-1) ** (n - i - 1) * t.unsqueeze(-1) ** i \
               + comb * i * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** (i - 1)
        return phi.float(), torch.clamp(dphi, -1e4, 1e4).float()

    # ---------- 3D Bernstein basis Φ(p) ∈ ℝ^{n³} ----------
    def basis(self, p, deriv=False):
        N = len(p)
        p = ((p - self.domain_min) / (self.domain_max - self.domain_min)).reshape(-1)
        phi, dphi = self._bernstein_1d(p, deriv)
        phi = phi.view(N, 3, self.n_func)
        px, py, pz = phi[:, 0, :], phi[:, 1, :], phi[:, 2, :]
        phi_xy = torch.einsum('ij,ik->ijk', px, py).view(-1, self.n_func ** 2)
        phi_xyz = torch.einsum('ij,ik->ijk', phi_xy, pz).view(-1, self.n_func ** 3)
        if not deriv: return phi_xyz, None
        dphi = dphi.view(N, 3, self.n_func)
        dx, dy, dz = dphi[:, 0, :], dphi[:, 1, :], dphi[:, 2, :]
        dpx = torch.einsum('ij,ik->ijk', torch.einsum('ij,ik->ijk', dx, py).view(-1, self.n_func ** 2), pz)
        dpy = torch.einsum('ij,ik->ijk', torch.einsum('ij,ik->ijk', px, dy).view(-1, self.n_func ** 2), pz)
        dpz = torch.einsum('ij,ik->ijk', phi_xy, dz).view(-1, self.n_func ** 3)
        return phi_xyz, torch.cat((dpx.unsqueeze(-1), dpy.unsqueeze(-1), dpz.unsqueeze(-1)), -1)

    # ------------------------- 训练主流程 -------------------------
    def train(self, stl_path, epochs=200, near=10_000, random=10_000):
        mesh = trimesh.load(stl_path)
        offset = mesh.bounding_box.centroid
        scale = np.max(np.linalg.norm(mesh.vertices - offset, axis=1))
        mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)          # 归一化到单位球

        # --- 生成 near / random 采样 ---
        near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(
            mesh,
            number_of_points=near,
            sign_method='normal',         # ✅ 不用渲染器
            surface_point_method='sample' # ✅ 直接在网格面上随机取点
        )

        # 2️⃣ 均匀随机采样（far / random）
        random_points, random_sdf = mesh_to_sdf.sample_sdf_near_surface(
            mesh,
            number_of_points=random,
            sign_method='normal',          # 不依赖 OpenGL
            surface_point_method='sample',
        )
        random_sdf[random_sdf < -1] *= -1                      # 同原论文对远处负值取绝对

        near_points = torch.from_numpy(near_points).to(self.device).float()
        near_sdf    = torch.from_numpy(near_sdf).to(self.device).float()
        random_points = torch.from_numpy(random_points).to(self.device).float()
        random_sdf    = torch.from_numpy(random_sdf).to(self.device).float()

        wb = torch.zeros(self.n_func ** 3, device=self.device)
        B  = torch.eye(self.n_func ** 3, device=self.device) / 1e-4

        for it in range(epochs):
            #   每轮随机抽取 mini-batch，加速/减少显存
            idx_n = torch.randint(near,   (1024,), device=self.device)
            idx_r = torch.randint(random, (256,),  device=self.device)
            p  = torch.cat((near_points[idx_n],   random_points[idx_r]), 0)
            sd = torch.cat((near_sdf[idx_n], random_sdf[idx_r]), 0)
            φ, _ = self.basis(p, deriv=False)

            K  = B @ φ.T @ torch.linalg.inv(torch.eye(len(p), device=self.device) + φ @ B @ φ.T)
            B -= K @ φ @ B
            wb += K @ (sd - φ @ wb)

        model = {'weights': wb.cpu(), 'offset': torch.from_numpy(offset), 'scale': scale}
        torch.save(model, self.model_path / f"object_BP_{self.n_func}.pt")
        print("✅ 训练完成并保存模型 →", self.model_path / f"object_BP_{self.n_func}.pt")
        return model

    # -------------------------- Marching-Cubes 可视化 --------------------------
    def reconstruct_mesh(self, model, nb=128, out="recon.stl", show=True):
        w  = model['weights'].to(self.device)
        dom = torch.linspace(self.domain_min, self.domain_max, nb, device=self.device)
        gx, gy, gz = torch.meshgrid(dom, dom, dom)
        p = torch.cat((gx.reshape(-1, 1), gy.reshape(-1, 1), gz.reshape(-1, 1)), 1).float()
        # 分块推理防 OOM
        chunks, sdf = torch.split(p, 10000), []
        for c in chunks:
            φ, _ = self.basis(c, False)
            sdf.append((φ @ w).cpu())
        sdf = torch.cat(sdf).view(nb, nb, nb).numpy()
        verts, faces, *_ = skimage.measure.marching_cubes(sdf, 0.0,
                          spacing=np.array([(self.domain_max - self.domain_min) / nb] * 3))
        mesh = trimesh.Trimesh(verts - 1, faces)
        if show: mesh.show()
        mesh.export(out)
        print("🖨️  已保存重建网格 →", out)

# ==================================== CLI ====================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stl", required=True, help="Path to the *.stl model to train")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--n_func", type=int, default=8, help="Bernstein basis order (n)")
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()

    bp = BPSDF(n_func=args.n_func, domain_min=-1.0, domain_max=1.0, device=args.device)
    mdl = bp.train(args.stl, epochs=args.epochs)
    bp.reconstruct_mesh(mdl, nb=128, out=CUR_DIR / "output_mesh.stl")
