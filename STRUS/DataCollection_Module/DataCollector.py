import yaml
import itertools as it
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch
class DataCollector():
    def __init__(self,configure_path,trajectory_generator,Attacker):
        self.TGenerator = trajectory_generator
        self.Attacker = Attacker
        self.load_config(configure_path)

        self.sigma = 1.0  
        self.alpha = 0.8  
     
    def load_config(self,config_path):
        f = open(config_path,"r",encoding="UTF-8")
        config = yaml.load(f,Loader=yaml.FullLoader) 
        f.close()

        self.trajectory_start_point_num = config["tarjectory_start_point_num"]
        self.time_step = config["time_step"] 
        self.time_interval = self.TGenerator.trajectory_info.period / self.trajectory_start_point_num 
        self.attack_types = config["attack_types"] 
        self.attack_aim_num = config["attack_aim_num"] 
        self.attack_aim_type = config["attack_aim_type"] 
        self.attack_aim_lst = sorted(list(it.combinations( 
            list(range(self.TGenerator.UAV_info.uav_num)),
            self.attack_aim_num)
        ))[:self.attack_aim_type] 
        if 3 in self.attack_types:
            self.real_state = []
            for i in range(self.attack_aim_type):
                self.real_state.append([])
                for j in range(self.trajectory_start_point_num):
                    self.real_state[i].append([])
        return


    def get_nxt_desired_position(self):
        nxt_desired_position = []
        for at_i in range(len(self.attack_types)):
            nxt_desired_position.append([])
            for am_i in range(self.attack_aim_type):
                nxt_desired_position[-1].append([])
                for sp_i in range(self.trajectory_start_point_num):
                    t = self.TGenerator.clock + self.TGenerator.trajectory_info.delta_t * sp_i
                    temp = self.TGenerator.get_swarm_state_by_t(t)[:,[1,2,3]]
                    nxt_desired_position[-1][-1].append(temp)
        self.nxt_desired_position = np.array(nxt_desired_position)
        return


    def init_desired_real_state(self):
        self.desired_state = []
        for at_i in range(len(self.attack_types)):
            self.desired_state.append([])
            for aa_i in range(self.attack_aim_type):
                self.desired_state[at_i].append([])
                for sp_i in range(self.trajectory_start_point_num):
                    self.desired_state[at_i][aa_i].append([self.SW[-1, aa_i, sp_i, 0].copy(), self.SW[-1, aa_i, sp_i, 1].copy()])

        self.real_state = []
        for aa_i in range(self.attack_aim_type):
            self.real_state.append([])
            for sp_i in range(self.trajectory_start_point_num):
                self.real_state[aa_i].append([self.SW[-1, aa_i, sp_i, 0].copy(), self.SW[-1, aa_i, sp_i, 1].copy()])

    def update_desired_state(self):
        for at_i in range(len(self.attack_types)):
            for aa_i in range(self.attack_aim_type):
                for sp_i in range(self.trajectory_start_point_num):
                    t = self.desired_state[at_i][aa_i][sp_i][-1][0,0] + self.TGenerator.trajectory_info.delta_t
                    self.desired_state[at_i][aa_i][sp_i].append(self.TGenerator.get_swarm_state_by_t(t))

    def init_sliding_window(self):
        self.reset()
        self.SW = []
        for sp_i in range(self.trajectory_start_point_num):
            t = self.TGenerator.clock + sp_i * self.time_interval
            window = []
            for ts_i in range(self.time_step):
                window.append(self.TGenerator.get_swarm_state_by_t(t+ts_i*self.TGenerator.trajectory_info.delta_t))
            self.SW.append(window)

        self.TGenerator.clock += self.TGenerator.trajectory_info.delta_t * self.time_step 

        self.SW = self.arr_repeat(np.array(self.SW),(len(self.attack_types),self.attack_aim_type))

        self.init_desired_real_state() 
        targets = self.add_attack_to_SW()


        return targets

    def update_sliding_window(self):
        nxt_time_step = []
        for sp_i in range(self.trajectory_start_point_num):
            t = self.TGenerator.clock + sp_i * self.time_interval
            nxt_time_step.append(self.TGenerator.get_swarm_state_by_t(t)) 
        self.TGenerator.clock_update()
        nxt_time_step = self.arr_repeat(np.array(nxt_time_step),(len(self.attack_types),self.attack_aim_type)) 
        self.SW[:, :, :, :-1] = self.SW[:, :, :, 1:]
        self.SW[:, :, :, -1] = nxt_time_step

        return

    def add_attack_to_SW(self,compensated_indexes=None):


        self.get_nxt_desired_position()  
        targets = self.Attacker.add_attack(  
            self.SW,  
            self.desired_state, self.real_state,
            self.nxt_desired_position,
            self.attack_types,self.attack_aim_lst,self.trajectory_start_point_num,
            self.TGenerator.get_single_state_by_command,
            compensated_indexes
        )
        self.Attacker.update_attacker() 
        self.update_desired_state()
        return targets

    def arr_repeat(self,arr,repeat_tuple):
        out = arr
        for i in range(len(repeat_tuple)-1,-1,-1): 
            out = [out]
            for r in range(repeat_tuple[i]-1):
                out.append(out[-1].copy()) 
            out = np.array(out)
        return np.array(out)

    def reset(self):
        self.TGenerator.reset_clock()
        return

    def _compute_euclidean_distance(self, positions):

        return squareform(pdist(positions))

    def _compute_adjacency_matrix(self, positions):

        sigma = self.sigma

        dist_matrix = self._compute_euclidean_distance(positions)
        adjacency = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

        np.fill_diagonal(adjacency, 0)
        # print(adjacency)
        return adjacency

    def _smooth_adjacency_matrix(self, current_adjacency):

        if self.previous_adjacency is None:
            smoothed = current_adjacency
        else:
            smoothed = self.alpha * self.previous_adjacency + (1 - self.alpha) * current_adjacency

        self.previous_adjacency = smoothed.copy()
        return smoothed

    def _compute_laplacian_matrix(self, adjacency):


        degree = np.sum(adjacency, axis=1)
        D = np.diag(degree)


        degree_sqrt_inv = np.zeros_like(degree)
        mask = degree > 0
        degree_sqrt_inv[mask] = 1.0 / np.sqrt(degree[mask])
        D_sqrt_inv = np.diag(degree_sqrt_inv)

        # : L = I - D^(-1/2) A D^(-1/2)
        identity = np.eye(adjacency.shape[0])
        laplacian = identity - D_sqrt_inv @ adjacency @ D_sqrt_inv

        lambda_max = 2.0 
        laplacian = 2 * laplacian / lambda_max - identity

        return laplacian

    def calc_gso(self, dir_adj, gso_type):

        try:
            n_vertex = dir_adj.shape[0]

            if sp.issparse(dir_adj) == False:
                dir_adj = sp.csc_matrix(dir_adj)
            elif dir_adj.format != 'csc':
                dir_adj = dir_adj.tocsc()

            id = sp.identity(n_vertex, format='csc')

            adj = 0.5 * (dir_adj + dir_adj.T)

            if np.any(np.isnan(adj.data)) or np.any(np.isinf(adj.data)):
                adj.data = np.nan_to_num(adj.data, nan=0.0, posinf=0.0, neginf=0.0)

            if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
                    or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
                adj = adj + id

            if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
                    or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
                row_sum = adj.sum(axis=1).A1

                epsilon = 1e-8
                row_sum = np.where(row_sum <= 0, epsilon, row_sum)
                row_sum = np.nan_to_num(row_sum, nan=epsilon, posinf=1.0, neginf=epsilon)

                row_sum_inv_sqrt = np.power(row_sum, -0.5)
                row_sum_inv_sqrt = np.nan_to_num(row_sum_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)

                deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
                sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

                if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
                    sym_norm_lap = id - sym_norm_adj
                    gso = sym_norm_lap
                else:
                    gso = sym_norm_adj

            elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
                    or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
                row_sum = np.sum(adj, axis=1).A1

                epsilon = 1e-8
                row_sum = np.where(row_sum <= 0, epsilon, row_sum)
                row_sum = np.nan_to_num(row_sum, nan=epsilon, posinf=1.0, neginf=epsilon)

                row_sum_inv = np.power(row_sum, -1)
                row_sum_inv = np.nan_to_num(row_sum_inv, nan=0.0, posinf=0.0, neginf=0.0)

                deg_inv = sp.diags(row_sum_inv, format='csc')
                rw_norm_adj = deg_inv.dot(adj)

                if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
                    rw_norm_lap = id - rw_norm_adj
                    gso = rw_norm_lap
                else:
                    gso = rw_norm_adj

            else:
                raise ValueError(f'{gso_type} is not defined.')

            return gso

        except Exception as e:
            print(f"计算GSO时出错: {e}")
            return sp.identity(dir_adj.shape[0], format='csc')

    def calc_chebynet_gso(self, gso):
        try:
            if sp.issparse(gso) == False:
                gso = sp.csc_matrix(gso)
            elif gso.format != 'csc':
                gso = gso.tocsc()

            id = sp.identity(gso.shape[0], format='csc')
            eigval_max = 2.0
            gso = 2 * gso / eigval_max - id
            if np.any(np.isnan(gso.data)) or np.any(np.isinf(gso.data)):
                print("警告: 切比雪夫GSO包含异常值，使用单位矩阵")
                return id

            return gso

        except Exception as e:
            print(f"计算切比雪夫GSO时出错: {e}")
            return sp.identity(gso.shape[0], format='csc')

    def generate_dynamic_graph_sequence(self, swarm_states):
        device = torch.device('cuda')
        shape = swarm_states.shape  # [attack_type, aim_type, start_point, time_step, N, F]
        # print(shape)
        time_step = shape[3]  
        N = shape[4]  
  
        for at_i in range(shape[0]):
            for aa_i in range(shape[1]):
                for sp_i in range(shape[2]):
                    self.previous_adjacency = None  
                    gso_sequence = []  

                    for ts_i in range(time_step): 
                        positions = swarm_states[at_i, aa_i, sp_i, ts_i, :, 0:3]
                        current_adj = self._compute_adjacency_matrix(positions)
                        smoothed_adj = self._smooth_adjacency_matrix(current_adj)
                        adj_sparse = sp.csr_matrix(smoothed_adj)
                        sym_norm_adj = self.calc_gso(adj_sparse, 'sym_norm_adj')
                        id_sp = sp.identity(N, format='csc')
                        laplacian_sparse = id_sp - sym_norm_adj.tocsc()
                        cheb_input = self.calc_chebynet_gso(laplacian_sparse)
                        gso = torch.from_numpy(cheb_input.toarray().astype(np.float32)).to(device)
                        gso_sequence.append(gso)  

                    gso_sequence = torch.stack(gso_sequence, dim=0)
                    if at_i == 0 and aa_i == 0 and sp_i == 0:
                        self.gso = gso_sequence  
        return