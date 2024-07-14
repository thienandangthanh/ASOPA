import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
from functools import reduce

from conf import args
from utils import *
# from network.data_generate import *
from data_generate import *
from pointer_network import *
from my_net import *
from network.pointer_network_2 import NeuralCombOptRL


def val_duibi(model, val_dataloader, use_qf=True, use_qs=False):
    if args.user_num<=5:
        use_qs=True
    reward_map=defaultdict(list)
    time_map=defaultdict(int)
    for batch_data in val_dataloader:
        users_g = batch_data.cpu().numpy()
        batch_data = batch_data.float().to(device).unsqueeze(1)
        t_start=time.time()
        _, _, _, pointers = model(batch_data)
        pointers = np.asarray([t.numpy() for t in pointers]).transpose()
        reward_map[0].extend(get_reward(users, users_g, pointers))
        time_map[0]+=time.time()-t_start
        duibi_ffs=[duibi_g_order_asc,duibi_g_order_desc]
        if use_qf:
            duibi_ffs.append(duibi_heuristic_method_qian)
        if use_qs:
            duibi_ffs.append(duibi_exhaustive_search)
        for i,ff in enumerate(duibi_ffs):
            i+=1
            time_start=time.time()
            for t in tqdm(batch_data):
                set_users_g(users, t[0])
                _,tt=ff(users,alpha=args.alpha)
                reward_map[i].append(tt)
            time_map[i]+=time.time()-time_start
    return reduce(lambda x,y:x.__add__(y),[[reward_map[i],time_map[i]] for i in range(5)])


# 固定随机种子
seed_everything(args.seed)
# 生成拓扑
users = generate_topology(args.user_num, args.d_min, args.d_max, args.w_min, args.w_max)
# 构建训练集与验证集
train_dataloader = get_dataloader(users, data_num=args.train_data_num, batch_size=args.train_batch_size, shuffle=True)
val_dataloader = get_dataloader(users, data_num=args.val_data_num, batch_size=args.val_batch_size, shuffle=False)
# 定义网络
# model=get_pointer_network(embedding_dim=args.embedding_dim,hidden_dim=args.hidden_dim,use_cuda=args.use_cuda)
model = NeuralCombOptRL()
device = 'cuda' if args.use_cuda else 'cpu'
# 与训练相关的
lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(args.epoch_num):
    # 训练
    model.train()
    for batch_i, batch_data in enumerate(tqdm(train_dataloader)):
        users_g = batch_data.cpu().numpy()
        batch_data = batch_data.float().to(device).unsqueeze(1)
        # probs,pointers=model(batch_data)
        _, probs, _, pointers = model(batch_data)
        # reward = get_reward(users, batch_data.cpu().numpy()/1000, pointers.cpu().numpy())
        pointers = np.asarray([t.numpy() for t in pointers]).transpose()
        print(pointers)
        reward = get_reward(users, users_g, pointers)
        if batch_i == 0:
            reward_mean = reward.mean()
            print(f'reward.mean()={reward.mean()}')
        else:
            print(f'epoch={epoch},reward_mean={reward_mean},reward.mean()={reward.mean()}')
            reward_mean = reward_mean * args.reward_beta + (1 - args.reward_beta) * reward.mean()
        advantage = torch.tensor(reward - reward_mean).to(device)
        # advantage = torch.tensor(reward - 55.9).to(device)
        # nn.Transformer
        logprobs = 0
        # nll=0
        reinforce = 0
        for prob, advant in zip(probs, advantage):
            # logprob=torch.log(prob)
            # logprob[(logprob<-1000).detach()]=0.
            logprob = torch.log(prob)
            logprobs += logprob
            # a,b=torch.max(logprob,dim=1)
            # a=torch.sum(a)
            # a=torch.prod(a)
            # if a<-1000:
            #     continue
            # logprob=prob
            # reinforce+=a*advant
            # nll+=-logprob
            # logprobs+=logprob
        # nll[(nll!=nll).detach()]=0.
        # logprobs[(logprobs<-1000).detach()]=0.
        # reinforce=advantage*logprobs
        reinforce = advantage * logprobs
        loss = -reinforce.mean()
        # loss=reinforce/len(probs)
        # print(f'loss={loss}')
        optimizer.zero_grad()
        loss.backward()
        # print(list(model.named_parameters()))
        # break
        # nn.utils.clip_grad_norm(model.parameters(),1,norm_type=2)

        optimizer.step()

        # break
        # if batch_i == 1:
        #     break
    # break
    lr /= 2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# print(model(batch_data))
# for name, parmas in model.named_parameters():
#     print(f'name:{name}')
#     print(f'para:{parmas}')
#     print(f'grad_value:{parmas.grad}')
# torch.save(model.state_dict(),time.strftime('%Y-%m-%d %H:%M:%S')+f'user_num_{args.user_num}_epoch_{epoch}.pth')
torch.save(model.state_dict(),f'user_num_{args.user_num}_epoch_{epoch}.pth')

reward,t_1, t_asc,t_2, t_desc,t_3, t_qf, t_4,t_qs,t_5=val_duibi(model,val_dataloader)
print(f'我们的方法:{sum(reward)/len(reward)},耗时{t_1}')
print(f'信道质量升序:{sum(t_asc)/len(reward)},耗时{t_2}')
print(f'信道质量降序:{sum(t_desc)/len(reward)},耗时{t_3}')
print(f'qian的启发式方法:{sum(t_qf)/len(reward)},耗时{t_4}')
print(f'穷搜的方法:{sum(t_qs)/len(reward)},耗时{t_5}')
