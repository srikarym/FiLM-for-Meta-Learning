# FiLM for Meta Learning and Reinforcement Learning

We have extended the work done in Model Agnostic meta learning by introducing FiLM: Feature-wise Linear Modulation layers for classification and reinforcement learning. It is efficient because gradiant updates (second order) is done on fewer parameters. The model is partitioned into two parts: context parameters (FiLM layers) that are adapted to individual tasks and shared parameters that are common across all the tasks. We show emperically that our approach achieves performance close to MAML in few shot classification. In case of Reinforcment Learning(Retro Contest) all the models in the contest used the same parameters across the different training levels we tried differentiating between them with a very few FiLM parameters.

Further details of Meta Learning and Reinforcement Learning Models are there in their respective folders.

Thank you for Ethan Perez and Rob Fergus for the support and idea to use FiLM in these contexts.
