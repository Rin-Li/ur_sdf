# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


import torch
import os
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import sys
sys.path.append(os.path.join(CUR_DIR,'../../RDF'))
from ur3e.ur3e import URRobot
from RDF.ur_rdf import BPSDF
from torchmin import minimize
import time
import math
import copy

PI = math.pi
NUMBER_OF_JOINT = 6  # number of joints in the robot

class DataGenerator():
    def __init__(self,device):
        # panda model
        self.UR = URRobot(device=device)
        self.bp_sdf = BPSDF(8,-1.0,1.0, self.UR,device)
        self.model = torch.load(os.path.join(CUR_DIR,'../models/BP_8.pt'), weights_only=False)
        self.q_max = self.UR.theta_max
        self.q_min = self.UR.theta_min
        # device
        self.device = device

        # data generation
        self.workspace = [[-0.5,-0.5,-0.5],[0.5,0.5,0.5]]
        self.n_disrete = 20         # total number of x: n_discrete**3
        self.batchsize = 1    # batch size of q
        # self.pose = torch.eye(4).unsqueeze(0).to(self.device).expand(self.batchsize,4,4).float()
        self.epsilon = 1e-3         # distance threshold to filter data

    def compute_sdf(self,x,q,return_index = False):
        # x : (Nx,3)
        # q : (Nq,6)
        # return_index : if True, return the index of link that is closest to x
        # return d : (Nq)
        # return idx : (Nq) optional

        pose = torch.eye(4).unsqueeze(0).to(self.device).expand(len(q),4,4).float()
        if not return_index:
            d,_ = self.bp_sdf.get_whole_body_sdf_batch(x,pose, q,self.model,use_derivative =False)
            d = d.min(dim=1)[0]
            return d
        else:
            d,_,idx = self.bp_sdf.get_whole_body_sdf_batch(x,pose, q,self.model,use_derivative =False,return_index = True)
            d,pts_idx = d.min(dim=1)
            idx = idx[torch.arange(len(idx)),pts_idx]
            return d,idx

    def given_x_find_q(self,x,q = None, batchsize = None,return_mask = False,epsilon = 1e-3):
        # x : (N,3)
        # scale x to workspace
        if not batchsize:
            batchsize = self.batchsize

        def cost_function(q):
            #  find q that d(x,q) = 0
            # q : B,2
            # x : N,3
            d = self.compute_sdf(x,q)
            cost = torch.sum(d**2)
            return cost
        
        t0 = time.time()
        # optimizer for data generation
        if q is None:
            q = torch.rand(batchsize,NUMBER_OF_JOINT).to(self.device)*(self.q_max-self.q_min)+self.q_min
        q0 = copy.deepcopy(q)
        
        try:
            res = minimize(
                cost_function, 
                q, 
                method='l-bfgs', 
                options=dict(line_search='strong-wolfe'),
                max_iter=50,
                disp=0
                )

            d,idx = self.compute_sdf(x,res.x,return_index=True)
            
            # 确保 d 和 idx 至少是 1 维张量
            if d.dim() == 0:
                d = d.unsqueeze(0)
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)
                
            d,idx = d.squeeze(),idx.squeeze()
            
            # 再次检查维度，如果仍然是 0 维，说明只有一个元素
            if d.dim() == 0:
                d = d.unsqueeze(0)
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)

            mask = torch.abs(d) < epsilon
            boundary_mask = ((res.x > self.q_min) & (res.x < self.q_max)).all(dim=1)
            final_mask = mask & boundary_mask
            
            if final_mask.any():
                final_q = res.x[final_mask]
                final_idx = idx[final_mask]
            else:
                # 如果没有找到有效解，返回空张量
                final_q = torch.empty(0, NUMBER_OF_JOINT, device=self.device)
                final_idx = torch.empty(0, dtype=torch.long, device=self.device)
            
        except Exception as e:
            print(f"优化过程出错: {e}")
            # 返回空结果
            final_q = torch.empty(0, NUMBER_OF_JOINT, device=self.device)
            final_idx = torch.empty(0, dtype=torch.long, device=self.device)
        
        if return_mask:
            return final_mask if 'final_mask' in locals() else torch.tensor([], dtype=torch.bool), final_q, final_idx
        else:
            return final_q, final_idx
        

    def distance_q(self,x,q):
        # x : (Nx,3)
        # q : (Np,6)
        # return d : (Np) distance between q and x in C space. d = min_{q*}{L2(q-q*)}. sdf(x,q*)=0

        # compute d
        Np = q.shape[0]
        q_template,link_idx = self.given_x_find_q(x)
        print(q_template.shape)

        if link_idx.min() == 0:
            return torch.zeros(Np).to(self.device)
        else:
            # UR3e 有 7 个 link (base + 6 joints)，但只有 6 个关节
            # link_idx 范围应该是 0-6 (对应 base, shoulder, upperarm, forearm, wrist1, wrist2, wrist3)
            link_idx = torch.clamp(link_idx, 0, 6)  # 确保索引在有效范围内
            
            d = torch.inf*torch.ones(Np, 6).to(self.device)  # 只有6个关节
            for i in range(1, 7):  # link 1-6 对应 6 个关节
                mask = (link_idx == i)
                if mask.any():
                    d_norm = torch.norm(q[:,:i].unsqueeze(1) - q_template[mask][:,:i].unsqueeze(0), dim=-1)
                    d[:,i-1] = torch.min(d_norm, dim=-1)[0]
        
        d = torch.min(d, dim=-1)[0]

        # compute sign of d
        d_ts = self.compute_sdf(x,q)
        mask = (d_ts < 0)
        d[mask] = -d[mask]
        return d

    def projection(self,x,q):
        q.requires_grad = True
        d = self.distance_q(x,q)
        grad = torch.autograd.grad(d,q,torch.ones_like(d),create_graph=True)[0]
        q_new = q - grad*d.unsqueeze(-1)
        return q_new

    def generate_offline_data(self,save_path = CUR_DIR):
        
        x = torch.linspace(self.workspace[0][0],self.workspace[1][0],self.n_disrete).to(self.device)
        y = torch.linspace(self.workspace[0][1],self.workspace[1][1],self.n_disrete).to(self.device)
        z = torch.linspace(self.workspace[0][2],self.workspace[1][2],self.n_disrete).to(self.device)
        x,y,z = torch.meshgrid(x,y,z, indexing='ij')
        pts = torch.stack([x,y,z],dim=-1).reshape(-1,3)
        
        total_points = len(pts)
        print(f"开始处理 {total_points} 个点 ({self.n_disrete}³ 网格)")
        print(f"工作空间: {self.workspace}")
        print(f"批次大小: {self.batchsize}")
        print(f"阈值: {self.epsilon}")
        print("=" * 80)
        
        data = {}
        start_time = time.time()
        valid_points = 0
        total_solutions = 0
        
        for i,p in enumerate(pts):
            point_start = time.time()
            
            try:
                q,idx = self.given_x_find_q(p.unsqueeze(0)) 
                data[i] = {
                    'x':    p.detach().cpu().numpy(),
                    'q':    q.detach().cpu().numpy(),
                    'idx':  idx.detach().cpu().numpy(),
                }
                
                if len(q) > 0:
                    valid_points += 1
                    total_solutions += len(q)
                
                point_time = time.time() - point_start
                
                # 计算进度信息
                progress = (i + 1) / total_points * 100
                elapsed_time = time.time() - start_time
                avg_time_per_point = elapsed_time / (i + 1)
                eta = avg_time_per_point * (total_points - i - 1)
                
                # 每10个点或每1%显示一次进度
                if (i + 1) % max(1, total_points // 100) == 0 or (i + 1) % 10 == 0:
                    print(f"点 {i+1:4d}/{total_points} ({progress:5.1f}%) | "
                            f"有效点: {valid_points:4d} | 总解数: {total_solutions:5d} | "
                            f"本点解数: {len(q):2d} | "
                            f"用时: {point_time:4.1f}s | "
                            f"剩余: {eta/60:4.1f}min | "
                            f"成功率: {valid_points/(i+1)*100:4.1f}%")
                
            except Exception as e:
                print(f"✗ 点 {i+1} 处理失败: {e}")
                data[i] = {
                    'x': p.detach().cpu().numpy(),
                    'q': np.array([]),
                    'idx': np.array([]),
                }
        
        total_time = time.time() - start_time
        print("=" * 80)
        print(f"数据生成完成！")
        print(f"总处理时间: {total_time/60:.2f} 分钟")
        print(f"平均每点用时: {total_time/total_points:.2f} 秒")
        print(f"有效点数: {valid_points}/{total_points} ({valid_points/total_points*100:.1f}%)")
        print(f"总解数: {total_solutions}")
        print(f"平均每个有效点的解数: {total_solutions/max(1,valid_points):.1f}")
        
        # 保存数据
        save_file = os.path.join(save_path, 'data.npy')
        np.save(save_file, data)
        print(f"数据已保存到: {save_file}")


def analysis_data(x):
    # Compute the squared Euclidean distance between each row
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    diff = diff.pow(2).sum(-1)

    # Set the diagonal elements to a large value to exclude self-distance
    diag_indices = torch.arange(x.shape[0])
    diff[diag_indices, diag_indices] = float('inf')
    
    # Compute the Euclidean distance by taking the square root
    diff = diff.sqrt()
    min_dist = torch.min(diff,dim=1)[0]
    print(f'distance\tmax:{min_dist.max()}\tmin:{min_dist.min()}\taverage:{min_dist.mean()}')



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"使用设备: {device}")

    # print("初始化 DataGenerator...")
    # gen = DataGenerator(device)
    # print("DataGenerator 初始化完成")
    
    # # 测试单个点，使用很小的批次大小
    # print("=" * 50)
    # print("测试 link_idx 的范围")
    # test_point = torch.tensor([[0.0, 0.0, 0.5]]).to(device)
    
    # try:
    #     # 使用很小的批次大小进行测试，先看看能不能跑通
    #     print("使用批次大小 10 进行测试...")
    #     q, idx = gen.given_x_find_q(test_point, batchsize=10)
    #     print(f"找到的 q 数量: {len(q)}")
    #     print(f"q 的形状: {q.shape}")
    #     print(f"idx 的形状: {idx.shape}")
    #     print(f"idx 的值: {idx}")
    #     print(f"idx 的最小值: {idx.min()}")
    #     print(f"idx 的最大值: {idx.max()}")
    #     print(f"idx 的唯一值: {torch.unique(idx)}")
        
    #     print(f"UR 机器人 link_order: {gen.UR.__class__.__module__}")
    #     print(f"NUMBER_OF_JOINT: {NUMBER_OF_JOINT}")
        
    #     # 如果成功，再试试稍大一点的
    #     if len(q) > 0:
    #         print("\n" + "=" * 50)
    #         print("测试批次大小 100...")
    #         q2, idx2 = gen.given_x_find_q(test_point, batchsize=100)
    #         print(f"找到的 q 数量: {len(q2)}")
    #         print(f"idx 的唯一值: {torch.unique(idx2)}")
        
    # except Exception as e:
    #     print(f"测试失败: {e}")
    #     import traceback
    #     traceback.print_exc()
    gen = DataGenerator(device)
    gen.generate_offline_data()