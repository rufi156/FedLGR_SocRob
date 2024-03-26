# Feature Aggregation with Latent Generative Replay for Federated Continual Learning of Socially Appropriate Robot Behaviours

## Overview
This is a PyTorch-based code implementation accompanying the iROS 2024 submission [Federated Feature Aggregation with Latent Generative Replay for Continually Learning Socially Appropriate Robot Behaviours](#). 
### Abstract
For widespread real-world applications, it is beneficial for robots to explore Federated Learning (FL) settings where several robots, deployed in parallel, can learn independently while also sharing their learning with each other. 
This work explores a simulated living room environment where robots need to learn the social appropriateness of their actions. We propose Federated Root (FedRoot), a novel weight aggregation strategy which disentangles feature learning across clients from individual task-based learning. 
Adapting popular FL strategies to use FedRoot instead, we present a novel FL benchmark for learning the social appropriateness of different robot actions in diverse social configurations. 
FedRoot-based methods offer competitive performance compared to others while offering sizeable (up to 86% for CPU usage and up to 72% for GPU usage) reduction in resource consumption. 
Furthermore, real-world interactions require social robots to dynamically adapt to changing environmental and task settings. To facilitate this, we propose Federated Latent Generative Replay (FedLGR), a novel Federated Continual Learning (FCL) strategy that uses FedRoot-based weight aggregation and embeds each client with a generator model for pseudo-rehearsal of learnt feature embeddings to mitigate forgetting in a resource-efficient manner. 
Our benchmark results demonstrate that FedRoot-based FCL methods outperform other methods while also offering sizeable (up to 84% for CPU usage and up to 92% for GPU usage) reduction in resource consumption, with FedLGR providing the best results across evaluations. 

## Table of Contents


- [Installation](#installation)
- [Data](#dataset)
- [Training](#training)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation

Ensure you have the necessary dependencies installed. Run the following command to set up the environment:

```bash
pip install -r requirements.txt
```

## Dataset

Access to the Manners-DB data files can be requested here: https://github.com/jonastjoms/MANNERS-DB.
The csv file with the labels, once acquired, should be placed under ```Data/all_data.csv```. Currently, a dummy file is included for reference.
All images should be placed under ```Data/images/```. Currently, an ```empty.file``` is placed there for reference, please remove that before running your code.

## Training

### Federated Learning
FL strategies are implemented as python packages. To execute the code on MANNERS-DB run the following:

```bash
bash run_FL_local.sh
```
Make sure all necessary paths are provided correctly. 

### Federated Continual Learning
FCL strategies are alsoimplemented as python packages. To execute the code on MANNERS-DB run the following:

```bash
bash run_CL_local.sh
```
Make sure all necessary paths are provided correctly. 


## Citation

```
@misc{Churamani2024Feature,  
  author        = {N. {Churamani} and S. {Checker} and H.T.L {Chiang} and H. {Gunes}},  
  title         = {{Feature Aggregation with Latent Generative Replay for Federated Continual Learning of Socially Appropriate Robot Behaviours}},   
  year          = {2024},  
  eprint        = {},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
 }
```

## Acknowledgement
**Funding:** [S. Checker](https://www.sakshamchecker.com) is with Delhi Technological University, India, and contributed to this work while undertaking a remote visiting studentship at the Department of Computer Science and Technology, University of Cambridge. 
[N. Churamani](https://nchuramani.github.io) and [H. Gunes](https://www.cl.cam.ac.uk/~hg410/) are with the Department of Computer Science and Technology, University of Cambridge, UK.
[H.T.L. Chiang](https://sites.google.com/view/lewispro/home) is with Google DeepMind.
This work is supported by Google under the GIG funding scheme. The authors also thank Jie Tan and Carolina Parada for their valuable feedback.

**Open Access:** For open access purposes, the authors have applied a Creative Commons Attribution (CC BY) licence to any Author Accepted Manuscript version arising.

**Data Access Statement:** This study involves secondary analyses of the existing datasets, that are described and cited in the text. 

