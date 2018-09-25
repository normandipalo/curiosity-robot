# An exploration of curiosity in robots.

In this project I explored the behaviour that emerges when training the Fetch robot in OpenAI's Gym environments with curiosity intrinsic reward. The robot learns to **discover the cube** by itself and rapidly starts playing with it, since it receives the highest curiosity reward from it. 
The RL algorithm is Proximal Policy Optimization, implemented in **TensorFlow Eager**.

## Requirements
You'll need a recent version of TensorFlow (>= 1.8 in order to use the Eager Execution).
The environments are based on the **MuJoCo** physics simulation, controlled through **OpenAI gym**. You can find a guide on how to install it in the OpenAI repository.

## Code structure
In *dyn_network.py* there is all the code relative to the predictive dynamical model, that learns to predict the next state given the current state and action. It also implements a normalizer, used to obtain a better performance. In *ppo.py* the main reinforcement learning algorithm  is implemented using Eager Execution. The notebook contains the main code to run the experiments.

## Contacts
Feel free to drop me a line if you find bugs or you just feel curious. You can find me on Twitter: @normandipalo.
