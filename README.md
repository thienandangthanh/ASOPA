
The code of ASOPA.
## Cite this work
L. Huang, B. Zhu, R. Nan, K. Chi and Y. Wu, "Attention-Based SIC Ordering and Power Allocation for Non-Orthogonal Multiple Access Networks," in IEEE Transactions on Mobile Computing, vol. 24, no. 2, pp. 939-955, Feb. 2025.
```
@ARTICLE{10700682,
  author={Huang, Liang and Zhu, Bincheng and Nan, Runkai and Chi, Kaikai and Wu, Yuan},
  journal={IEEE Transactions on Mobile Computing}, 
  title={Attention-Based SIC Ordering and Power Allocation for Non-Orthogonal Multiple Access Networks}, 
  year={2025},
  volume={24},
  number={2},
  pages={939-955},
  keywords={NOMA;Interference cancellation;Resource management;Optimization;Wireless networks;Wireless communication;Uplink;Complexity theory;Measurement;Heuristic algorithms;Deep reinforcement learning (DRL);resource allocation;successive interference cancellation (SIC);non-orthogonal multiple access (NOMA)},
  doi={10.1109/TMC.2024.3470828}}
```
## About authors
* [Liang Huang](https://scholar.google.com/citations?user=NifLoZ4AAAAJ) ,lianghuang@zjut.edu.cn
* [Bincheng Zhu](https://ieeexplore.ieee.org/author/37089420307) ,bczhu@zjut.edu.cn
* [Nanrun Kai](https://ieeexplore.ieee.org/author/37089596991) ,rknan@zjut.edu.cn
* [Kaikai Chi](https://scholar.google.com/citations?user=MrdiGtMAAAAJ&hl=en&oi=ao) ,kkchi@zjut.edu.cn
* [Yuan Wu](https://scholar.google.com/citations?hl=en&user=H1bxY_4AAAAJ) ,yuanwu@um.edu.mo

# Train
Run run.py, and ASOPA can be trained by a variable of users. <br>
The details of the dataset can be found in ./problems/problem_noop.py class NOOP_allnum_Daset.<br>
# Validation
Run ASOPA_validation.py, and the network utility and execution latency of ASOPA will be displayed. <br>
Run Baseline_validation.py, and the network utility and execution latency of baseline algorithms will be displayed.<br>
You can also change the val_user_num of conf.py and val_graph_size of options.py to see the simulation with 5 or 10 users.

## Configure
When you validate ASOPA and baseline algorithms with 8 users, you can set options.py and conf.py as follows.
### options.py:
![image](https://github.com/user-attachments/assets/90b90963-d782-484c-8012-2d6d3aacf8f2)
### conf.py:
![image](https://github.com/user-attachments/assets/2e928ced-fa09-4cef-9e36-b2856c192355)

## Example
We can obtain the simulation results as follows:
### ASOPA:
![image](https://github.com/user-attachments/assets/f3062470-f106-4437-99f8-747ddd77f9da)
### Baseline:
![image](https://github.com/user-attachments/assets/c682f269-1701-414c-a1cf-b183bfdb9908)

