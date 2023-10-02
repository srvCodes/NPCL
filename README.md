# NPCL: Neural Processes for Uncertainty-Aware Continual Learning

## What is NPCL? 

&#8594; NPCL is a probabilistic continual learning (CL) model that relies on the neural processes - a class of meta learners exploiting data-driven priors - to model task distributions.

## How does NPCL work?

&#8594; NPCL leverages a two-layered hierarchical latent variable model such that the top-layer latent $z^G$ models the global distribution of all seen tasks while the bottom-layered latent $z^t$ models the task-specific distributions conditioned on the global one. Below is a rough sketch of this:

![alt text](https://github.com/srvCodes/NPCL/blob/main/images/Picture1.png)

## Why uncertainty-based CL task modeling?

&#8594;  Besides achieving accuracies that are on par with the state-of-the-art deterministic models, NPCL offers the additional perks of lower model calibration error, enhanced few-shot replay performance, novel data identification capability, and instance-level model confidence evaluation [1]. Taking the lattermost example as a reference, here is a GIF of the Q-Q plots verifying the normality assumption of the top-two probability differences for NPCL:

![alt text](https://github.com/srvCodes/NPCL/blob/main/images/normality_check.gif)


References: 

[1] Han, X., Zheng, H., & Zhou, M. (2022). CARD: Classification and Regression Diffusion Models. ArXiv, abs/2206.07275.
