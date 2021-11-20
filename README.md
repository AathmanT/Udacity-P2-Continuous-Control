# Project 2 Continuous Control
This is the second project in the Udacity Deep Reinforcement Learning Nano Degree Program

### Project Details
For this project, the goal is to train an agent (double-jointed arm) to maintain its position at the target
location for as many time steps as possible.

<img src="assets/reacher.gif" alt="" title="Reacher Environment" />

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project we will use the environment containing a single agent

**Solving the Environment**  
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

### Getting Started
1. Follow the instructions in the [Udacity DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment.
2. Install tensorboardX and update numpy to support tensorboardX. tensorboardX can be used to visualize the Actor and Critic networks and the scores as and when the agent is training. This can help to tune hyper-parameters much easily and also helps to compare all the runs.
```
pip install tensorboardX==2.4
pip install numpy==1.19.5
```
3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

   (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

   (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip)
   to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

4. Place the file in your DRLND GitHub repository, in the `Udacity-P2-Continuous-Control/` folder, and unzip (or decompress) the file.

### Instructions

The `Continuous_Control.ipynb` contains instructions on how to explore the environment, train an agent, view its performance score and save and load it to visualize how the trained agent interacts with the environment.