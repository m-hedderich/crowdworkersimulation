
## Explaining Crowdworker Behaviour Through Computational Rationality

This is the supplementary material (preprint + code + models) for the paper ["Explaining crowdworker behaviour through computational rationality"](https://www.tandfonline.com/doi/full/10.1080/0144929X.2024.2329616).

You can find the pre-print of the paper [here](https://michael-hedderich.de/assets/pdf/ExplainingCrowdworkerComputationalRationality.pdf).

## Abstract
Crowdsourcing has transformed whole industries by enabling the collection of human input at scale. Attracting high quality responses remains a challenge, however. Several factors affect which tasks a crowdworker chooses, how carefully they respond, and whether they cheat. In this work, we integrate many such factors into a simulation model of crowdworker behaviour rooted in the theory of computational rationality. The root assumption is that crowdworkers are rational and choose to behave in a way that maximises their expected subjective payoffs. The model captures two levels of decisions: (i) a worker's choice among multiple tasks and (ii) how much effort to put into a task. We formulate the worker's decision problem and use deep reinforcement learning to predict worker behaviour in realistic crowdworking scenarios. We examine predictions against empirical findings on the effects of task design and show that the model successfully predicts adaptive worker behaviour with regard to different aspects of task participation, cheating, and task-switching. To support explaining crowdworker actions and other choice behaviour, we make our model publicly available.

## File Structure
The ``code`` directory contains the code of the RL platform as well as the tutorial
(as Jupyter notebook and PDF). The ``exp`` directory contains two trained RL models.

## Installation

The system has been tested on Linux with Python 3.8 and the libraries specified in the requirements.txt file. If you use conda, you can just install them via
```
conda create -n crowdworker_env python=3.8
conda activate crowdworker_env
pip install -r requirements.txt
pip install notebook ipywidgets # for the tutorial Juypter notebook
```
Make sure that you install the correct versions of Python and the libraries to ensure that everything works.


## Tutorial
The Jupyter notebook Crowdworker-Behavior_Tutorial.ipynb provides an extensive tutorial that shows how to run the crowdworking environment, how to analyze the behavior of the worker and how to train new worker agents. Just start Jupyter Notebook, open the provided tutorial notebook and go through it step by step.

For more detailed information, you can check the code which provides extensive comments.


## Trained Models
We provide the following trained models in the *exp* directory:

- *paper-exps_cheating_qa3-0.1_rep0.9* is the model used for replicating the effects A1-4 and B2. 
It was trained with cheating deterrents (gold questions and reputation system). 
- *paper-exps_cheating_qa3-0_rep0.9* is a model trained without cheating deterrents.

Each model directory contains a *config.json* file which specifies the training parameters
of the RL model. The files *user_properties.json* and *anti_cheat_settings.json* specify
the properties of the worker and the cheating deterrents used in this setting. The file
*task_properties_distributions.txt* specifies which type of task-giver distributions were used
in human-readable form (alternatively, a pickled file exists with the actual objects). The other
files are log-files from the training and the *model.save* file which contains the saved RL model.

## Citation
If you find this work useful, please consider citing it as
```
@article{hedderich2024ExplainingCrowdworker,
	author = {Michael A. Hedderich and Antti Oulasvirta},
    title = {Explaining crowdworker behaviour through computational rationality},
    journal = {Behaviour \& Information Technology},
    year = {2024},
    publisher = {Taylor \& Francis},
    doi = {10.1080/0144929X.2024.2329616},
    URL = { https://www.tandfonline.com/doi/abs/10.1080/0144929X.2024.2329616},
}
```

## Contact
If you have any questions or suggestions, feel free to reach out to the first author at mail ( at ) michael-hedderich.de or raise an issue on GitHub.

