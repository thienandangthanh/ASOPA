import argparse

parser=argparse.ArgumentParser()
# 用来生成拓扑的
parser.add_argument('--user_num',default=10,help='The number of users')
# 问题验证相关
parser.add_argument('--val_user_num',default=8,help='The number of users in Validation')
parser.add_argument('--val_epoch',default=300,help='The epoch of ASOPA for Validation')
# 1,2,1,10
parser.add_argument('--d_min',default=20,help='Minimum distance from user to base station, default 20')
parser.add_argument('--d_max',default=100,help='Maximum distance from user to base station, default 100')
parser.add_argument('--w_min',default=1,help='Minimum throughput weights for users') #1,5
parser.add_argument('--w_max',default=32,help='Maximum throughput weights for users')

parser.add_argument('--num_min',default=5,help='Minimum number of users in dataset') #1,5
parser.add_argument('--num_max',default=10,help='Maximum number of users in dataSET')


# 与问题相关的参数
parser.add_argument('--alpha',default=1)

parser.add_argument('--noise',default=3.981e-15,help='Gaussian white noise at the base station,default -184 dBm/Hz * 1MHz, i.e., 3.981e-16') #1e-8~1e-13
# 随机化相关
parser.add_argument('--seed',default=1234,help='Random seed')
# 与学习相关的
parser.add_argument('--frames_num',default=300000)
# 与数据集相关
parser.add_argument('--train_data_num',default=120000)
# parser.add_argument('--train_data_num',default=120,help='用来训练的数据量')
parser.add_argument('--val_data_num',default=1000,help='The number of samples for validation')

# 与pointer_network有关
parser.add_argument('--embedding_dim',default=2,help='The dimension of input embedding in pointer network')
parser.add_argument('--hidden_dim',default=2,help='The dimension of hidden layer in pointer network')
parser.add_argument('--use_cuda',default=False,help='Use gpu or not')
# 与训练有关
parser.add_argument('--epoch_num',default=2,help='The epoch of training')
parser.add_argument('--lr',default=1e-3,help='learning rate')
parser.add_argument('--reward_beta',default=0.9,help='Controls the average update rate of the rewards during training, the larger the update rate the slower it is.')



args=parser.parse_args()