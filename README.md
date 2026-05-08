# Under-Construction
Making an agent walk with legs using reinforcement learning in Unity was a real challenge for me. After several attempts, it finally worked with two main factors: radian/degree conversion and Evolution Strategies. Configurable Joint could not be used because it did not behave like a rigid connection, which made Articulation Body necessary. The previous problem with Articulation Body—no movement during deterministic actions—was caused by not converting between radians and degrees. Joint values (jointPosition) in the Articulation Body are represented in radians, so without converting them to degrees, the joint angles are nearly indistinguishable. This caused incorrect observations and resulted in no movement. However, after fixing this problem and training with PPO, it was difficult to observe any meaningful increase in the reward because the agent initially fell immediately and received very similar rewards regardless of its actions, making PPO’s clipped update mechanism ineffective for discovering meaningful walking behavior, since the small reward differences provided little useful gradient information for policy improvement. It is unclear whether PPO alone would work with long training time, but I wanted to see the reward increase within a short time. So I decided to implement ES to find initial neural network parameters capable of slight walking behavior and then continued training with PPO to further improve and stabilize locomotion.

ES and PPO are implemented in PyTorch, and ES is based on the OpenAI ES paper [*Evolution Strategies as a Scalable Alternative to Reinforcement Learning*](https://arxiv.org/pdf/1703.03864). For PPO, the previous implementation stopped collecting transitions when an episode terminated and waited for the other agents. This was modified so that transition collection continues even after episode termination, while ensuring that GAE is computed correctly. The observations and hyperparameters are partially based on the quadrupedal paper [*Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*](https://arxiv.org/pdf/2109.11978).


## Environment
### Unity
- Unity Editor: 6000.3.0f1
- ML Agents: 4.0.2
- Sentis: 2.5.0

### Python
- Python 3.10.12
