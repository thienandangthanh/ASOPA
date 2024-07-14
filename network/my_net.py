# import torch
import torch.nn as nn
import time

from conf import args
from my_utils import *
from resource_allocation_optimization import *
from network.pointer_network import get_pointer_network, PointerNet
from tqdm import tqdm
# from numba import jit


# @jit
def get_reward(users, g, decode_order, alpha=args.alpha, noise=args.noise):
    reward_list = []
    for t_g, t_decode_order in tqdm(zip(g, decode_order)):
        set_users_g(users, t_g)
        users_order = sort_by_decode_order(users, t_decode_order)
        reward = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha, noise=noise)
        reward_list.append(reward)
        # reward = get_max_sum_weighted_alpha_throughput(users=users_order, alpha=alpha, noise=noise,use_nlopt=True)
        # reward_list.append(reward)
    return np.asarray(reward_list)

# sl=100
# args.user_num=8
# users = generate_topology(user_number=args.user_num)
# random_set_users_g(users)
# # users_g=get_usrs_g(users)
# # x=torch.tensor(users_g).unsqueeze(0).float()
# users_g_hat=get_users_g_hat(users)
# users_g=np.random.random([sl,args.user_num])*users_g_hat
# x=torch.tensor(users_g).float()
# pointer_network=PointerNet(128,256)
# outputs,pointers=pointer_network(x)
# time_start=time.time()
# reward=get_reward(users,x.numpy(),pointers.numpy())
# time_end=time.time()
# print(f'在用户数为{args.user_num}时,计算{sl}次需要的时间为{time_end-time_start}s')
# print(f'reward={reward}')
