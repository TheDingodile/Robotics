---
layout: post
title:  "Weighing-My-Options"
date:   2024-05-26 23:12:58 +0900
categories: jekyll update
permalink: /Weighing-My-Options.html
---

**Introduction**

In robotics, the state of the environment is typically represented by what is visible. However, this approach can miss key environmental aspects, such as the mass distribution of objects, which is crucial for effective interaction. When humans grasp and carry objects, they naturally infer information about the mass distribution and adapt their grip to minimize torque, maximize stability, and ensure a successful grasp. Simply grasping at the volumetric center of a box is not sufficient; robots need to understand mass distribution to perform these tasks effectively.

Our project focuses on enhancing robotic grasping and carrying abilities by leveraging existing sensors (visual and haptic) to gather information about an object's mass-distribution. Our project explores these methods to enhance robot learning, enabling robots to better understand and manipulate objects.

**Our Approach**

*Dual-network structure*

*Reward engineering*

**Experimental Environment**

Here is an image from the simulation.

<img src="initial_state.png" alt="cube" width="400">


Here is a simple illustration of the mass-distribution of the cube.

![cube](CubeMassDistribution.png)

<div style="display: flex;">
    <img src="CubeMassDistribution.png" alt="CMD" style="width: 45%; margin-right: 10px;">
    <img src="BalanceBox.png" alt="Balance" style="width: 45%;">
</div>



For testing our method we setup a simulation in [Pybullet](https://pybullet.org/). We use the framework given in [MyGym](https://mygym.readthedocs.io/en/latest/) for our setup...

**Results**

it went well (i hope)

**Conclusion**

Nice job


**References**

- [Pybullet](https://pybullet.org/)

- [MyGym](https://mygym.readthedocs.io/en/latest/)

```








