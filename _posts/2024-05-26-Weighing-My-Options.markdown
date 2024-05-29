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

First we introduce some basic concepts that are used in our project.

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
L^{\text{Policy}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

where
- $$ r_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} $$ is the probability ratio,
- $$ \hat{A}_t $$ is the estimated advantage at time $$ t $$,
- $$ \epsilon $$ is a hyperparameter that controls the clipping range.

To calculate the advantage estimates we use a value head. This makes PPO an actor-critic method. The value head is trained by simply minimizing the mean squared error between the predicted value and the return-to-go. Formally we have:

$$
L^{\text{Value}} = \mathbb{E}_t \left[ (V_{\theta}(s_t) - R_t)^2 \right]
$$

where $$ R_t $$ is the return-to-go at time $$ t $$.

**Environment**

In this section we describe the environment in which our method is tested.

We aimed for 3 main objectives in our environment:

- Achievable for our agent within scope, compute, and time constraints
- Challenging enough to require mass-distribution information for solving
- Suitable to evaluate the effectiveness of our approach

We settled on the task of balancing a cube on a pole. The cube has a randomly sampled center of mass (COM) from inside the cube. To solve the task everytime, the agent has to figure out where the COM is and move that part towards the middle of the pole to balance the cube. The agent controls a robot arm to grasp and carry the cube.

The framework used is [MyGym](https://mygym.readthedocs.io/en/latest/) which is built on top of Pybullet. This framework provides a simple interface for creating custom environments and training agents.

Here is an image from the simulation.

<div style="text-align: center;">
    <img src="initial_state.png" alt="cube" width="300">
    <p style="font-style: italic; font-size: 0.8em;">Initial state of the environment. The goal is for the robot arm to grab the green cube, and put in on top of the black pole such that the cube is balanced.</p>
</div>


***Reward engineering***

MyGym comes with defaults reward functions which are designed to speed up training. In addition, we split the task into sub-tasks to guide the agent towards the final goal. The sub-tasks are:

- Grasp the cube
- Lift the cube
- Drop the cube on the ground (we explain this sub-task later)
- Drop the cube on the pole
- Balance the cube on the pole

**(check if this is correct)**

***State Representation***

The state is represented by a high-level description of the environment. It is described by the global 3D position and velocity of the box and robot arm **(check if this is correct)**. Formally, we have:

$$
s = (x_{\text{box}}, y_{\text{box}}, z_{\text{box}}, x_{\text{arm}}, y_{\text{arm}}, z_{\text{arm}}, v_{x_{\text{box}}}, v_{y_{\text{box}}}, v_{z_{\text{box}}}, v_{x_{\text{arm}}}, v_{y_{\text{arm}}}, v_{z_{\text{arm}}})
$$


Due to the Center of mass (COM) of the box not being visible in this definition, this state definition is suspected to be under-represented. Specifically, the Markov property is likely insufficiently satisfied for this state definition, as the COM impacts the best way to balance the cube on the pole.

Therefore, we introduce an augmented state where we include the COM of the object, described by it's relative position to the object's center. This is given by:

$$
s_{\text{COM}} = (s, x_{\text{COM}}, y_{\text{COM}}, z_{\text{COM}})
$$

We intent to figure out whether we can train a model to predict: 

$$
COM = (x_{\text{COM}}, y_{\text{COM}}, z_{\text{COM}})
$$

In particular, we want to investigate whether such an approach of calculating and utilizing the COM is more effective than having a network that deals with the non-markovian states by using the trajectory history as modelled by a LSTM.

***Action space***

The action space is a 3D continuous space, where the robot arm can move in the x, y, and z directions. The action space is defined as:

$$
a = (x_{\text{arm}}, y_{\text{arm}}, z_{\text{arm}})
$$

An under-the-hood control system is used to convert these actions into the robot arm's movements. Using this approach, the agent can focus on learning the high-level task of balancing the cube on the pole, rather than the low-level control of the robot arm.




**Methods**


We propose two directions to tackle this problem, which will be compared in our experiments:

***Approach 1: Backbone network***

The first idea is to have an LSTM backbone network with a policy head, a value head, and a COM head. The value head is used to do the PPO updates of the policy head, while the COM head simply is used to predict the COM of the object with MSE loss at each timestep. We introduce a weight to the loss of the COM head, to balance the importance of the different losses, formally we have:

$$
L = L^{\text{Policy}} + \lambda_{1} L^{\text{Value}} + \lambda_{2} L^{\text{COM}}
$$


where $$ \lambda $$ is a hyperparameter that controls the importance of the COM loss. In our experiments we use $$ \lambda_{1} = 0.5 $$ and $$ \lambda_{2} = 0.1 $$

The idea is that the LSTM can learn the trajectory history of the object, and use this to predict the COM. Then the hope is that due to enforcing the LSTM to pick up on the COM, the policy learn to use this information.

Below is a visualisation of the network architecture.

<div style="text-align: center;">
    <img src="BackBone.png" alt="The Cube" width="500">
    <p style="font-style: italic; font-size: 0.8em;">The network architecture of the Backbone Network approach. The policy-head outputs a 3-dimensional vector, the value-head outputs a single value, and the COM-head outputs a 3-dimensional vector also.</p>
</div>



***Approach 2: Dual-network structure***

In contrast to approach 1, this is a modular approach, where we intent to approximate the augmented state $$ s_{\text{COM}} $$, and then feed it to the PPO network.

If we have access to such augmented state, our state would closer approximate the Markov property, and we therefore hypothesise that we don't need an LSTM for the actor-critic network. Thus, instead, we can use a simple feed-forward network to approximate the policy and value functions. 

The COM prediction network is trained separately. In particular, we introduce a special sub-task (of picking up and dropping the box) to train the COM prediction network. The policy is first given this sub-task to lift up the cube and drop it. After this sub-task is completed, the COM predictor is given the trajectory of the sub-task as input and the COM as target. We use an LSTM for the COM prediction network. The hope is that the trajectory reveals information about the COM, and that the LSTM can pick up on this.

For the special (lift and drop) sub-task, when the COM network has not made a prediction yet, we set the COM to the center of the box. However, when the COM network has made a prediction, we fix the COM to the predicted value for the rest of that trajectory. This way, the policy network can learn to use the COM information.


Below is a visualisation of the network architecture.

<div style="text-align: center;">
    <img src="Dual-Network.png" alt="The Cube" width="800">
    <p style="font-style: italic; font-size: 0.8em;">The network architecture of the Dual-network approach. The COM-head outputs a 3-dimensional vector which is concatinated to the state and given to the actor-critic network.</p>
</div>

**Experiments**

**Results**

it went well (i hope)

**Limitations and Future Work**

Our work has several limitations and directions for future research:

1. ***Approach 1:***
   - In approach 1, we encourage the network to attend to the Center of Mass (COM) based on its trajectory rather than actively exploring it.
   - ***Recommendation:*** One extension could involve adding a reward for reducing COM prediction error. This would incentivize the policy to explore states that provide valuable information about the COM, potentially improving its ability to balance the cube.

2. ***Approach 2:***
   - Approach 2 fixes the COM after prediction, without allowing the COM network to predict at each timestep.
   - ***Recommendation:*** To enhance interaction and training, a future extension could involve having the COM network predict the COM dynamically at each timestep. This could lead to a more adaptive policy and better handling of COM variations during balancing.

3. ***Environmental Complexity:***
   - The current environment is relatively simple.
   - ***Recommendation:*** Future work should evaluate our methods in more complex environments to assess their robustness and scalability.

4. ***Task Specificity:***
   - Our sub-tasks are domain-specific and may not generalize well to other tasks.
   - ***Recommendation:*** Exploring ways to generalize the learning of COM to more diverse tasks could enhance the applicability and utility of our approach.





**Conclusion**

Nice job :smile: :happy:


**References**

- [Code](https://github.com/mschoene/myGym)

- [Pybullet](https://pybullet.org/)

- [MyGym](https://mygym.readthedocs.io/en/latest/)

```








