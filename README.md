
The code of ASOPA.
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

