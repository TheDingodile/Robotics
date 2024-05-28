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

## Methods

### Reinforcement Learning

The reinforcement learning problem is defined by a Markov Decision Process (MDP), which consists of:

- A set of states $$ S $$
- A set of actions $$ A $$
- A transition function $$ P(s' \mid s, a) $$
- A reward function $$ R(s, a) $$
- A discount factor $$ \gamma \in [0, 1] $$

The goal is to find a policy $$ \pi(a \mid s) $$ that maximizes the expected cumulative reward:

$$
\mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right]
$$

Reward in our case, sub-goals in our case

*State Representation*

The state is represented by a high-level description of our environment. It is described by the global 3D position and velocity of the box and robot arm **(check if this is correct)**. Formally, we have:

$$
\pi(s \mid a)
$$

$$
s = (x_{\text{box}}, y_{\text{box}}, z_{\text{box}}, x_{\text{arm}}, y_{\text{arm}}, z_{\text{arm}}, v_{x_{\text{box}}}, v_{y_{\text{box}}}, v_{z_{\text{box}}}, v_{x_{\text{arm}}}, v_{y_{\text{arm}}}, v_{z_{\text{arm}}})
$$


Due to the Center of mass (COM) of the box not being visible a part of this definition, and how the COM impacts the best way to balance the cube, the Markov property is insufficiently satisfied by this state definition.

Therefore, we introduce an augmented state where we include the COM of the object, described by it's relative position to the object's center. This is given by:

$$
s_{\text{COM}} = (s, x_{\text{COM}}, y_{\text{COM}}, z_{\text{COM}})
$$

We intent to figure out whether we can train a model to predict the COM of the object, and how this can be used to improve the robot's ability to balance the cube on the pole. In particular, we want to investigate whether such an approach of calculating and utilizing the COM is more effective than having a network that deals with the non-markovian states by using the trajectory history as modelled by a LSTM.



*PPO*

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








