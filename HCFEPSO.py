import numpy as np
import random
import multiprocessing
import time

# ======================全局超参======================
COMMUNICATION_ROUND_N = 20
CLIENT_PER_GROUP = 3
PARTICLE_PER_CLIENT = 40

W_START = 0.9
W_END = 0.4
C1 = 2.0
C2 = 2.0
C3 = 1.7
C4 = 0.8
V_MAX = np.array([3, 3, 15])

def demo_fitness(x):
    fit = np.sum(x**2)
    obj = np.array([fit, fit*0.1, fit*0.2, fit*0.3])
    return fit, obj


class MPSOClient:
    """矩阵粒子群MPSO客户端，单个客户端独立粒子群"""
    def __init__(self, dim, particle_num, bounds, sub_seed):
        self.dim = dim
        self.num_particles = particle_num
        self.bounds = np.array(bounds)  # shape(dim,2): [low,high]
        self.rng = np.random.default_rng(sub_seed)

        self.x = self.rng.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.num_particles, self.dim)
        )
        self.v = np.zeros_like(self.x)

        self.pbest = self.x.copy()
        self.pbest_fit = np.full(self.num_particles, np.inf)
        self.gbest_local = None
        self.gbest_local_fit = np.inf

        self.raw_obj_pop = None
        self.guide_elite_particle = None

    def calc_all_fitness(self, fitness_func):
        """传入适应度函数，返回fit数组，同时保存原始目标向量"""
        fit_list = []
        obj_list = []
        for xi in self.x:
            fit_val, obj_vec = fitness_func(xi)
            fit_list.append(fit_val)
            obj_list.append(obj_vec)
        self.raw_obj_pop = np.array(obj_list)
        return np.array(fit_list)

    def get_all_raw_objectives(self):
        raw_data = []
        if self.raw_obj_pop is None:
            return raw_data
        for idx in range(self.num_particles):
            raw_data.append({
                "particle_x": self.x[idx].copy(),
                "obj": self.raw_obj_pop[idx].copy()
            })
        return raw_data

    def matrix_pso_update(self, global_best_particle, fitness_func):
        w = W_START - (W_START - W_END) * (self.current_local_iter / self.max_local_iter)
        r1 = self.rng.random((self.num_particles, self.dim))
        r2 = self.rng.random((self.num_particles, self.dim))
        r3 = self.rng.random((self.num_particles, self.dim))
        r4 = self.rng.random((self.num_particles, self.dim))

        v_update = w * self.v
        v_update += C1 * r1 * (self.pbest - self.x)
        v_update += C2 * r2 * (self.gbest_local[None, :] - self.x)
        if global_best_particle is not None:
            v_update += C3 * r3 * (global_best_particle[None, :] - self.x)
        if self.guide_elite_particle is not None:
            v_update += C4 * r4 * (self.guide_elite_particle[None, :] - self.x)

        v_update = np.clip(v_update, -V_MAX[0], V_MAX[0])
        self.v = v_update

        self.x += self.v
        for d in range(self.dim):
            self.x[:, d] = np.clip(self.x[:, d], self.bounds[d,0], self.bounds[d,1])

        fit = self.calc_all_fitness(fitness_func)

        better_mask = fit < self.pbest_fit
        self.pbest[better_mask] = self.x[better_mask].copy()
        self.pbest_fit[better_mask] = fit[better_mask]

        min_fit_idx = np.argmin(self.pbest_fit)
        current_min_fit = self.pbest_fit[min_fit_idx]
        if current_min_fit < self.gbest_local_fit:
            self.gbest_local_fit = current_min_fit
            self.gbest_local = self.pbest[min_fit_idx].copy()
        return fit

    def run_local_iter(self, local_iters, global_best_particle, fitness_func):
        self.max_local_iter = local_iters
        min_idx = np.argmin(self.pbest_fit)
        self.gbest_local = self.pbest[min_idx].copy()
        self.gbest_local_fit = self.pbest_fit[min_idx]
        for self.current_local_iter in range(local_iters):
            self.matrix_pso_update(global_best_particle, fitness_func)
        return (self.gbest_local_fit,
                self.gbest_local,
                self.raw_obj_pop.copy(),
                self.x.copy(),
                self.pbest.copy(),
                self.pbest_fit.copy(),
                self.guide_elite_particle)


class HCFEPSOServer:
    """联邦服务器；通信回合为每个客户端生成专属指导精英粒子"""
    def __init__(self):
        self.clients = []
        self.guide_elite_map = dict()

    def add_client(self, client: MPSOClient):
        self.clients.append(client)

    def aggregate_info(self, all_client_raw_data, global_obj_min, global_obj_max):
        self.guide_elite_map.clear()
        for cid, particle_list in all_client_raw_data.items():
            best_F = float("inf")
            best_p = None
            for item in particle_list:
                obj = item["obj"]
                px = item["particle_x"]
                n1 = (obj[0] - global_obj_min[0])/(global_obj_max[0]-global_obj_min[0]+1e-10)
                n2 = (obj[1] - global_obj_min[1])/(global_obj_max[1]-global_obj_min[1]+1e-10)
                n3 = (obj[2] - global_obj_min[2])/(global_obj_max[2]-global_obj_min[2]+1e-10)
                n4 = (obj[3] - global_obj_min[3])/(global_obj_max[3]-global_obj_min[3]+1e-10)
                F = n1+n2+n3+n4
                if F < best_F:
                    best_F = F
                    best_p = px.copy()
            self.guide_elite_map[cid] = best_p


def client_local_wrapper(args):
    client, local_iters, global_best_particle, fitness_func = args
    res = client.run_local_iter(local_iters, global_best_particle, fitness_func)
    return res


def run_hcfepso_algorithm(dim, bounds, fitness_func, max_global_iter, exp_seed=0):
    """
    HCFEPSO算法入口
    """
    random.seed(exp_seed)
    main_ss = np.random.SeedSequence(exp_seed)
    total_client = CLIENT_PER_GROUP
    client_seeds = main_ss.spawn(total_client)

    server0 = HCFEPSOServer()
    all_clients = []
    for cid in range(total_client):
        sub_seed = client_seeds[cid]
        client = MPSOClient(dim, PARTICLE_PER_CLIENT, bounds, sub_seed)
        # 初始化适应度
        _ = client.calc_all_fitness(fitness_func)
        min_idx = np.argmin(client.pbest_fit)
        client.gbest_local = client.pbest[min_idx].copy()
        client.gbest_local_fit = client.pbest_fit[min_idx]
        server0.add_client(client)
        all_clients.append(client)

    global_best_fit = np.inf
    global_best_particle = None
    history = []

    def norm(val, vmin, vmax, eps=1e-10):
        delta = vmax - vmin
        if abs(delta) < eps:
            return 0.5
        return (val - vmin) / delta

    # 初始代采集目标
    init_all_obj = []
    for cli in all_clients:
        raw_list = cli.get_all_raw_objectives()
        for item in raw_list:
            init_all_obj.append(item["obj"])
    init_all_obj = np.array(init_all_obj)
    frozen_norm_min = np.min(init_all_obj, axis=0).copy()
    frozen_norm_max = np.max(init_all_obj, axis=0).copy()

    # 初代全局最优
    init_best_F = float("inf")
    init_best_p = None
    for cli in all_clients:
        raw_list = cli.get_all_raw_objectives()
        for item in raw_list:
            obj = item["obj"]
            px = item["particle_x"]
            n1 = norm(obj[0], frozen_norm_min[0], frozen_norm_max[0])
            n2 = norm(obj[1], frozen_norm_min[1], frozen_norm_max[1])
            n3 = norm(obj[2], frozen_norm_min[2], frozen_norm_max[2])
            n4 = norm(obj[3], frozen_norm_min[3], frozen_norm_max[3])
            F = n1 + n2 + n3 + n4
            if F < init_best_F:
                init_best_F = F
                init_best_p = px.copy()
    global_best_particle = init_best_p.copy()
    global_best_fit = init_best_F

    pool_size = min(total_client, 16)
    with multiprocessing.Pool(processes=pool_size) as pool:
        for global_it in range(max_global_iter):
            task_args = []
            feed_particle = global_best_particle
            for c in all_clients:
                task_args.append((c, 1, feed_particle, fitness_func))

            parallel_results = pool.map(client_local_wrapper, task_args)

            for idx, res in enumerate(parallel_results):
                fit, gbest_p, raw_obj_pop, pop_x, pbest_mat, pbest_fit_arr, guide_elite = res
                target_client = all_clients[idx]
                target_client.gbest_local_fit = fit
                target_client.gbest_local = gbest_p
                target_client.raw_obj_pop = raw_obj_pop
                target_client.x = pop_x
                target_client.pbest = pbest_mat
                target_client.pbest_fit = pbest_fit_arr

            all_obj_this_iter = []
            all_particle_this_iter = []
            client_raw_dict = dict()
            for cid, cli in enumerate(all_clients):
                raw_data_list = cli.get_all_raw_objectives()
                client_raw_dict[cid] = raw_data_list
                for item in raw_data_list:
                    all_obj_this_iter.append(item["obj"])
                    all_particle_this_iter.append(item["particle_x"])
            all_obj_this_iter = np.array(all_obj_this_iter)
            cur_global_min = np.min(all_obj_this_iter, axis=0)
            cur_global_max = np.max(all_obj_this_iter, axis=0)

            trigger_comm = (global_it % COMMUNICATION_ROUND_N == 0)
            if trigger_comm:
                frozen_norm_min = cur_global_min.copy()
                frozen_norm_max = cur_global_max.copy()
                server0.aggregate_info(client_raw_dict, cur_global_min, cur_global_max)
                for cid, guide_p in server0.guide_elite_map.items():
                    all_clients[cid].guide_elite_particle = guide_p

            best_global_F = float("inf")
            best_particle = None
            for px, obj_vec in zip(all_particle_this_iter, all_obj_this_iter):
                n1 = norm(obj_vec[0], frozen_norm_min[0], frozen_norm_max[0])
                n2 = norm(obj_vec[1], frozen_norm_min[1], frozen_norm_max[1])
                n3 = norm(obj_vec[2], frozen_norm_min[2], frozen_norm_max[2])
                n4 = norm(obj_vec[3], frozen_norm_min[3], frozen_norm_max[3])
                F_global = n1 + n2 + n3 + n4
                if F_global < best_global_F:
                    best_global_F = F_global
                    best_particle = px.copy()

            if best_global_F < global_best_fit:
                global_best_fit = best_global_F
                global_best_particle = best_particle.copy()

            history.append(global_best_fit)

    return {
        "best_fitness": global_best_fit,
        "best_particle": global_best_particle,
        "fitness_history": history
    }


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    DIM = 6
    BOUNDS = [[0,10]] * DIM
    result = run_hcfepso_algorithm(
        dim=DIM,
        bounds=BOUNDS,
        fitness_func=demo_fitness,
        max_global_iter=30,
        exp_seed=1234
    )
    print("best_fitness:", result["best_fitness"])
    print("best_particle:", result["best_particle"])
