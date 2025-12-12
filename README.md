# **G1 Humanoid RL – Isaac Lab Reinforcement Learning Project**

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Docs](https://img.shields.io/badge/docs-available-blue)
![License](https://img.shields.io/badge/license-BSD--3--Clause-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

This repository contains reinforcement-learning environments and training pipelines for the **Unitree G1 Humanoid** using **NVIDIA Isaac Lab**, **skrl**, and **PPO**.
The goal is to develop a scalable curriculum for full-body humanoid behaviors: **stand, walk, sit, get-up**, and more.

---

# **1. Technical Architecture**

The project is structured into four major layers: **Simulation → RL Interface → Algorithm → Policy Outputs**

```
┌──────────────────────────────────────────────────────────────┐
│                      SYSTEM ARCHITECTURE                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                        Isaac Lab Layer                        │
│  • GPU rigid-body simulation (PhysX)                          │
│  • G1 humanoid USD model                                      │
│  • Environment cloning (6000 envs)                            │
│  • Sensors: root state, joint states, gravity projection      │
└──────────────────────────────────────────────────────────────┘
                   │ Observations, rewards, resets
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                     RL Environment Wrapper                    │
│  • Custom env: G1TasksEnv                                     │
│  • Reward computation                                         │
│  • Curriculum configs (stand, walk, slow walk, legs-only)     │
│  • Action application (29-DOF control)                        │
└────────────────────────────────────────────────────────────────┘
                   │ batched rollouts
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                        RL Training Layer                      │
│  • skrl PPO implementation                                    │
│  • Rollout collection                                         │
│  • Normalization + advantage estimation                       │
│  • Policy + value networks                                    │
└──────────────────────────────────────────────────────────────┘
                   │ gradients
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                   Policy Network Inference                    │
│  • Predicts continuous joint targets                          │
│  • Scaled to joint space (±0.5 rad)                           │
└──────────────────────────────────────────────────────────────┘
```

---

# **2. PPO Architecture**

This project uses **Proximal Policy Optimization (PPO)** through **skrl**, optimized for large-scale parallel Isaac environments.

```
┌──────────────────────────────────────────────────────────────┐
│                       PPO ARCHITECTURE                        │
└──────────────────────────────────────────────────────────────┘

                     ┌──────────────────────────┐
                     │      Observations        │
                     │  (base vel, pos, joints) │
                     └─────────────┬────────────┘
                                   ▼
                         ┌───────────────────┐
                         │  Policy Network   │
                         │  πθ(a | s)        │
                         └───────┬───────────┘
                                 │ Actions
                                 ▼
                      (G1 Joint Targets Applied)
                                 ▼
                     Rewards, dones, next obs
                                 ▼
                    ┌─────────────────────────┐
                    │  Advantage Estimation   │
                    │      (GAE-Lambda)       │
                    └──────────┬──────────────┘
                               ▼
                        ┌──────────────┐
                        │ Value Network│
                        │    V(s)      │
                        └─────┬────────┘
                              ▼
                     ┌──────────────────────┐
                     │   PPO Update Step     │
                     │  L_clip = min(...)    │
                     │  entropy bonus        │
                     │  value loss           │
                     └──────────────────────┘
```

### Key PPO Features Used

* **Clip ratio:** prevents destructive policy updates
* **Entropy bonus:** maintains exploration
* **GAE:** smooth and low-variance advantage estimates
* **Batch optimization:** across thousands of simulated humanoids
* **Continuous action space:** 29-dimensional joint commands

---

# **3. Isaac Lab**

Isaac Lab provides:

* High-speed GPU physics (PhysX 5)
* Easy USD-based robot import (G1 model)
* Observation and reward APIs
* Environment cloning for 6000+ parallel scenes

Environment source files:

* `g1_tasks_env_cfg.py` (config) 
* `g1_tasks_env.py` (logic + rewards) 

---

# **4. Supported RL Frameworks**

## **4.1 skrl**

Primary library used for PPO training.

Advantages:

* Native Isaac Lab integration
* GPU vectorization
* Stable PPO implementation
* Clean Python API

## **4.2 RL-Games (optional planned integration)**

* Highly optimized PPO
* Used in many humanoid locomotion papers
* Good for long-horizon tasks (sit, get-up)

---

# **5. Tasks (Current + Upcoming)**

### **Current Tasks**

| Task                  | Purpose                | Config                 |
| --------------------- | ---------------------- | ---------------------- |
| **Standing Balance**  | Keep upright, no drift | `G1StandEnvCfg`        |
| **Slow Walking**      | Learn stable stepping  | `G1SlowWalkEnvCfg`     |
| **Dynamic Walking**   | Full gait at 1.0 m/s   | `G1TasksEnvCfg`        |
| **Legs-Only Walking** | Learn fundamental gait | `G1WalkLegsOnlyEnvCfg` |

### **Planned Tasks**

* Sit down
* Stand up
* Get-up (from ground)
* Turning / side-stepping
* Variable-speed locomotion
* Recovery behaviors

---

# **6. Installation**

```
pip install -r requirements.txt
pip install skrl
```

Install Isaac Lab according to NVIDIA’s documentation.

---

# **7. Training**

Run PPO training:

```
python train_ppo.py --task g1_walk --num_envs 6000
```

---

# **8. Repository Structure**

```
.
├── README.md
├── scripts
│   ├── list_envs.py
│   ├── random_agent.py
│   ├── skrl
│   │   ├── play.py
│   │   └── train.py
│   └── zero_agent.py
└── source
    └── g1_tasks
        ├── config
        │   └── extension.toml
        ├── docs
        │   └── CHANGELOG.rst
        ├── g1_tasks
        │   ├── __init__.py
        │   ├── tasks
        │   │   ├── direct
        │   │   │   ├── g1_tasks
        │   │   │   │   ├── agents
        │   │   │   │   │   ├── __init__.py
        │   │   │   │   │   └── skrl_ppo_cfg.yaml
        │   │   │   │   ├── g1_tasks_env_cfg.py
        │   │   │   │   ├── g1_tasks_env.py
        │   │   │   │   ├── __init__.py
        │   │   │   ├── __init__.py
        │   │   ├── __init__.py
        │   └── ui_extension_example.py
        ├── pyproject.toml
        └── setup.py
```

---

# **9. License**

BSD-3 Clause (matching Isaac Lab).

