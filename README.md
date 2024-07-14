
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
![image](https://github.com/user-attachments/assets/afeb6af4-35e5-4fb0-a377-694348485ca9)

### conf.py:
![image](https://github.com/user-attachments/assets/05d5d50d-1970-4246-9d59-f67006a9e1d2)

## Example
We can obtain the simulation results as follows:
### ASOPA:
![image](https://github.com/user-attachments/assets/f3062470-f106-4437-99f8-747ddd77f9da)
### Baseline:
![image](https://github.com/Jil-Menzerna/ASOPA/assets/62533692/ed8e6576-bd8c-4098-91fb-e42908488c9a) 
