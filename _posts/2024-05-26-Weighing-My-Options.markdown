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

<div style="text-align: center;">
    <img src="initial_state.png" alt="cube" width="300">
    <p style="font-style: italic; font-size: 0.8em;">Initial state of the environment. The goal is for the robot arm to grab the green cube, and put in on top of the black pole such that the cube is balanced.</p>
</div>


Here is a simple illustration of the mass-distribution of the cube.

<div style="text-align: center;">
    <img src="BalanceCube.png" alt="The Cube" width="500">
    <p style="font-style: italic; font-size: 0.8em;">The Center of Mass (COM) is randomly sampled (uniformly) from inside the cube. This means to balance the cube on the pole it has to figure out (from playing around with the cube) where the COM is and move that part towards the middle of the pole.</p>

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








