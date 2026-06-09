# Bimanual Imitation Learning Benchmark

This repository contains experiments for bimanual imitation learning using `robosuite` and `robomimic`. The main focus is training and evaluating policies on the `TwoArmTransport` task using Behavior Cloning, BC-RNN, and Diffusion Policy.

## 1. Environment Setup

Create and activate the conda environment:

```bash
conda activate robosuite
```

Install robomimic in editable mode:

```bash
cd ~/Downloads/robomimic
pip install -e .
```

## 2. Data Preparation

Download the official robomimic Transport dataset:

```bash
python robomimic/scripts/download_datasets.py \
  --tasks transport \
  --dataset_types ph \
  --hdf5_types low_dim
```

Check the dataset:

```bash
python robomimic/scripts/get_dataset_info.py \
  --dataset datasets/transport/ph/low_dim_v15.hdf5
```

The dataset contains 200 demonstrations and 93,752 transitions for the `TwoArmTransport` task.

## 3. Data Variations / Physical Perturbations

To evaluate robustness, physical properties of the transported object can be modified during evaluation.

Perturbations include:

* Object mass / weight
* Object friction
* Center of gravity shift

Example perturbation settings:

```python
mass_scale = 2.0
friction_scale = 1.0
com_shift = (0.0, 0.0, 0.0)
```

These perturbations test whether the trained policy can generalize under dynamics changes.

## 4. Training

### Behavior Cloning

Train feedforward BC:

```bash
python robomimic/scripts/train.py \
  --config two_arm_bc.json \
  --dataset datasets/transport/ph/low_dim_v15.hdf5
```

### BC-RNN

Train recurrent BC with temporal context:

```bash
python robomimic/scripts/train.py \
  --config two_arm_bc_rnn_seq20.json \
  --dataset datasets/transport/ph/low_dim_v15.hdf5
```

### Diffusion Policy

Train Diffusion Policy:

```bash
python robomimic/scripts/train.py \
  --config two_arm_diffusion.json \
  --dataset datasets/transport/ph/low_dim_v15.hdf5
```

## 5. Evaluation

Evaluate a trained checkpoint:

```bash
python robomimic/scripts/run_trained_agent.py \
  --agent /path/to/checkpoint.pth \
  --n_rollouts 20 \
  --horizon 800
```

Render one rollout:

```bash
python robomimic/scripts/run_trained_agent.py \
  --agent /path/to/checkpoint.pth \
  --n_rollouts 1 \
  --horizon 800 \
  --render
```

Save rollout video:

```bash
python robomimic/scripts/run_trained_agent.py \
  --agent /path/to/checkpoint.pth \
  --n_rollouts 1 \
  --horizon 800 \
  --video_path /tmp/rollout.mp4
```

## 6. Current Results

| Method                     | Task            | Success Rate |
| -------------------------- | --------------- | ------------ |
| Feedforward BC             | TwoArmTransport | 0.0          |
| BC-RNN seq10               | TwoArmTransport | 0.2          |
| BC-RNN seq20               | TwoArmTransport | 0.5          |
| Diffusion Policy epoch 500 | TwoArmTransport | 0.6          |

## 7. Notes

Large files are excluded from GitHub, including:

* datasets
* trained model checkpoints
* rollout videos

These files should be stored locally or uploaded separately if needed.


# robomimic

<p align="center">
  <img width="24.0%" src="docs/images/task_lift.gif">
  <img width="24.0%" src="docs/images/task_can.gif">
  <img width="24.0%" src="docs/images/task_tool_hang.gif">
  <img width="24.0%" src="docs/images/task_square.gif">
  <img width="24.0%" src="docs/images/task_lift_real.gif">
  <img width="24.0%" src="docs/images/task_can_real.gif">
  <img width="24.0%" src="docs/images/task_tool_hang_real.gif">
  <img width="24.0%" src="docs/images/task_transport.gif">
 </p>

[**[Homepage]**](https://robomimic.github.io/) &ensp; [**[Documentation]**](https://robomimic.github.io/docs/introduction/overview.html) &ensp; [**[Study Paper]**](https://arxiv.org/abs/2108.03298) &ensp; [**[Study Website]**](https://robomimic.github.io/study/) &ensp; [**[ARISE Initiative]**](https://github.com/ARISE-Initiative)

-------
## Latest Updates
- [06/20/2025] **v0.5.0**: Diffusion Policy, multi-dataset training, language-conditioned policies, and more! 
- [03/11/2025] **v0.4.0**: support for [robosuite v1.5](https://github.com/ARISE-Initiative/robosuite/tree/v1.5.1) and migrate robomimic datasets to HuggingFace
- [10/11/2023] **v0.3.1**: support for extracting, training on, and visualizing depth observations for robosuite datasets
- [07/03/2023] **v0.3.0**: BC-Transformer and IQL :brain:, support for DeepMind MuJoCo bindings :robot:, pre-trained image reps :eye:, wandb logging :chart_with_upwards_trend:, and more
- [05/23/2022] **v0.2.1**: Updated website and documentation to feature more tutorials :notebook_with_decorative_cover:
- [12/16/2021] **v0.2.0**: Modular observation modalities and encoders :wrench:, support for [MOMART](https://sites.google.com/view/il-for-mm/home) datasets :open_file_folder: [[release notes]](https://github.com/ARISE-Initiative/robomimic/releases/tag/v0.2.0) [[documentation]](https://robomimic.github.io/docs/v0.2/introduction/overview.html)
- [08/09/2021] **v0.1.0**: Initial code and paper release

-------

## Colab quickstart
Get started with a quick colab notebook demo of robomimic without installing anything locally.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b62r_km9pP40fKF0cBdpdTO2P_2eIbC6?usp=sharing)


-------

**robomimic** is a framework for robot learning from demonstration.
It offers a broad set of demonstration datasets collected on robot manipulation domains and offline learning algorithms to learn from these datasets.
**robomimic** aims to make robot learning broadly *accessible* and *reproducible*, allowing researchers and practitioners to benchmark tasks and algorithms fairly and to develop the next generation of robot learning algorithms.

## Core Features

<p align="center">
  <img width="50.0%" src="docs/images/core_features.png">
 </p>

<!-- **Standardized Datasets**
- Simulated and real-world tasks
- Multiple environments and robots
- Diverse human-collected and machine-generated datasets

**Suite of Learning Algorithms**
- Imitation Learning algorithms (BC, BC-RNN, HBC)
- Offline RL algorithms (BCQ, CQL, IRIS, TD3-BC)

**Modular Design**
- Low-dim + Visuomotor policies
- Diverse network architectures
- Support for external datasets

**Flexible Workflow**
- Hyperparameter sweep tools
- Dataset visualization tools
- Generating new datasets -->


## Reproducing benchmarks

The robomimic framework also makes reproducing the results from different benchmarks and datasets easy. See the [datasets page](https://robomimic.github.io/docs/datasets/overview.html) for more information on downloading datasets and reproducing experiments.

## Docker

You can use the `Dockerfile` to easily build a containerized environment for setting up robomimic with Python 3.9, Miniconda, robosuite, and PyTorch (CPU/GPU support).

To build, run:
`docker build -t robomimic .`

To run without GPU (CPU only), run:
`docker run -it robomimic`

To run with GPU (if available), run:
`docker run --gpus all -it robomimic`

## Troubleshooting

Please see the [troubleshooting](https://robomimic.github.io/docs/miscellaneous/troubleshooting.html) section for common fixes, or [submit an issue](https://github.com/ARISE-Initiative/robomimic/issues) on our github page.

## Contributing to robomimic
This project is part of the broader [Advancing Robot Intelligence through Simulated Environments (ARISE) Initiative](https://github.com/ARISE-Initiative), with the aim of lowering the barriers of entry for cutting-edge research at the intersection of AI and Robotics.
The project originally began development in late 2018 by researchers in the [Stanford Vision and Learning Lab](http://svl.stanford.edu/) (SVL).
Now it is actively maintained and used for robotics research projects across multiple labs.
We welcome community contributions to this project.
For details please check our [contributing guidelines](https://robomimic.github.io/docs/miscellaneous/contributing.html).

## Citation

Please cite [this paper](https://arxiv.org/abs/2108.03298) if you use this framework in your work:

```bibtex
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Ajay Mandlekar and Danfei Xu and Josiah Wong and Soroush Nasiriany and Chen Wang and Rohun Kulkarni and Li Fei-Fei and Silvio Savarese and Yuke Zhu and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2021}
}
```
