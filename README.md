# FedLGR_SocRob
Repository for submission on Federated Feature Aggregation with Latent Generative Replay for Continually Learning Socially Appropriate Robot Behaviours


## Installation

Ensure you have the necessary dependencies installed. Run the following command to set up the environment:

```bash
pip install -r requirements.txt
```

## Dataset

Access to the Manners-DB data files can be requested here: https://github.com/jonastjoms/MANNERS-DB/tree/master, once acquired, the data should be placed under ```Data```.

## Training

The repository is divided into two parts
- Federated Learning
- Federated Continual Learning

### Federated Learning
FL strategies are implementad as python packages. To execute the code on MANNERS-DB run the following:

```bash
bash run_FL_local.sh
```

### Federated Contiinual Learning
FCL strategies are implementad as python packages. To execute the code on MANNERS-DB run the following:

```bash
bash run_CL_local.sh
```



## Citation


## Acknowledgement
**Funding:** [S. Checker](https://www.sakshamchecker.com) is with Delhi Technological University, India, and contributed to this work while undertaking a remote visiting studentship at the Department of Computer Science and Technology, University of Cambridge. [N. Churamani](https://nchuramani.github.io) and [H. Gunes](https://www.cl.cam.ac.uk/~hg410/) are with the Department of Computer Science and Technology, University of Cambridge, UK
H.T.L. Chiang is with Google DeepMind.
This work is supported by Google under the GIG funding scheme. The authors also thank Jie Tan and Carolina Parada for their valuable feedback.

**Open Access:** For open access purposes, the authors have applied a Creative Commons Attribution (CC BY) licence to any Author Accepted Manuscript version arising.

**Data Access Statement:** This study involves secondary analyses of the existing datasets, that are described and cited in the text. 

