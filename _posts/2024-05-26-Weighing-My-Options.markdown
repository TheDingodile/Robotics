---
layout: post
title:  "Weighing-My-Options"
date:   2024-05-26 23:12:58 +0900
categories: jekyll update
permalink: /Weighing-My-Options.html
---

**Introduction**

In robotics, the state of the environment is typically represented by visually present features. However, this approach can miss key environmental aspects, such as the mass distribution of objects, which is crucial for effective interaction. When humans grasp and carry objects, they naturally infer information about the mass distribution and adapt their grip to minimize torque, maximize stability, and ensure a successful grasp. Simply grasping at the volumetric center of a box is not sufficient; robots need to understand mass distribution to perform these tasks effectively.

Our project focuses on enhancing robotic grasping and carrying abilities by leveraging existing sensors to gather information about an object's mass-distribution. Our project explores these methods to enhance robot learning, enabling robots to better understand and manipulate objects.

To test our method we created a simulation in [Pybullet](https://pybullet.org/), where a robot arm must balance a cube on a pole. Below is a visualisation of this situation.

<div style="text-align: center;">
    <img src="BalanceCube.png" alt="The Cube" width="500">
    <p style="font-style: italic; font-size: 0.8em;">The Center of Mass (COM) is randomly sampled (uniformly) from inside the cube. This means to balance the cube on the pole it has to figure out (eg. from playing around with the cube) where the COM is and move that part towards the middle of the pole.</p>

</div>

**Preliminaries**

First we introduce some concepts that are important for our project.

***Reinforcement Learning***

We can model the problem of balancing the cube on the pole as a reinforcement learning problem. The reinforcement learning problem is defined by a Markov Decision Process (MDP), which consists of:

- A set of states $$ S $$
- A set of actions $$ A $$
- A transition function $$ P(s' \mid s, a) $$
- A reward function $$ R(s, a) $$
- A discount factor $$ \gamma \in [0, 1] $$

The goal is to find a policy $$ \pi(a \mid s) $$ that maximizes the expected cumulative reward:

$$
\mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right]
$$



The Markov property is assumed for an MDP, which formally mean the current state contains all the information needed to make a decision. Formally we have that:

$$
P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots)
$$

***Policy Proximal Optimisation (PPO)***

For training our agents we opt to use PPO. This is a simple state-of-the-art policy gradient method that is widely used, and easy to implement. The objective function is:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

where
- $$ r_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} $$ is the probability ratio,
- $$ \hat{A}_t $$ is the estimated advantage at time $$ t $$,
- $$ \epsilon $$ is a hyperparameter that controls the clipping range.

To calculate the advantage estimates we need value head. This makes PPO an actor-critic method.

**Environment**

In this section we describe the environment in which our method is tested.

We aimed for 3 main objectives in our environment:

- Achievable for our agent within scope, compute, and time constraints
- Challenging enough to require mass-distribution information for solving
- Suitable to evaluate the effectiveness of our approach

We settled on the task of balancing a cube on a pole. The cube has a randomly sampled center of mass (COM) from inside the cube. To solve the task everytime, the agent has to figure out where the COM is and move that part towards the middle of the pole to balance the cube. The agent controls a robot arm to grasp and carry the cube.

The framework used is [MyGym](https://mygym.readthedocs.io/en/latest/) which is built on top of Pybullet. This framework provides a simple interface for creating custom environments and training agents.

***Reward engineering***

MyGym comes with defaults reward functions which are designed to speed up training. In addition, we split the task into sub-tasks to guide the agent towards the final goal. The sub-tasks are:

- Grasp the cube
- Lift the cube
- Drop the cube on the ground (we explain this sub-task later)
- Drop the cube on the pole

**(check if this is correct)**



**Methods**

***State Representation***

The state is represented by a high-level description of our environment. It is described by the global 3D position and velocity of the box and robot arm **(check if this is correct)**. Formally, we have:

$$
s = (x_{\text{box}}, y_{\text{box}}, z_{\text{box}}, x_{\text{arm}}, y_{\text{arm}}, z_{\text{arm}}, v_{x_{\text{box}}}, v_{y_{\text{box}}}, v_{z_{\text{box}}}, v_{x_{\text{arm}}}, v_{y_{\text{arm}}}, v_{z_{\text{arm}}})
$$


Due to the Center of mass (COM) of the box not being visible a part of this definition, and how the COM impacts the best way to balance the cube, this state definition is suspected to be under-represented. Specifically, the Markov property is likely insufficiently satisfied for this state definition.

Therefore, we introduce an augmented state where we include the COM of the object, described by it's relative position to the object's center. This is given by:

$$
s_{\text{COM}} = (s, x_{\text{COM}}, y_{\text{COM}}, z_{\text{COM}})
$$

We intent to figure out whether we can train a model to predict the COM of the object, and how this can be used to improve the robot's ability to balance the cube on the pole. In particular, we want to investigate whether such an approach of calculating and utilizing the COM is more effective than having a network that deals with the non-markovian states by using the trajectory history as modelled by a LSTM.


*Network architectures*


**Our Approach**

We propose two directions to tackle this problem:

*Modularized Dual-network structure*

The first idea is to train two seperate networks: an observation network and a policy networks. 




First we define a

The policy is first given the subtask to lift up the cube and drop it.

The perception network is responsible for extracting information about the mass distribution of the object, while the control network is responsible for generating the robot's actions. The perception network takes in sensor data and outputs a representation of the object's mass distribution. The control network then uses this representation to generate actions that are more effective at grasping and carrying the object.




*Dual-network structure*

*Reward engineering*

**Experimental Environment**

Here is an image from the simulation.

<div style="text-align: center;">
    <img src="initial_state.png" alt="cube" width="300">
    <p style="font-style: italic; font-size: 0.8em;">Initial state of the environment. The goal is for the robot arm to grab the green cube, and put in on top of the black pole such that the cube is balanced.</p>
</div>


For testing our method we setup a simulation in [Pybullet](https://pybullet.org/). We use the framework given in [MyGym](https://mygym.readthedocs.io/en/latest/) for our setup...

**Results**

it went well (i hope)

**Conclusion**

Nice job :smile: :happy:


**References**

- [Pybullet](https://pybullet.org/)

- [MyGym](https://mygym.readthedocs.io/en/latest/)

```








