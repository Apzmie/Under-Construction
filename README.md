# Under-Construction
Making an agent walk with legs using reinforcement learning in Unity was a real challenge for me. After several attempts, it finally worked with two main factors: radian/degree conversion and Evolution Strategies. Configurable Joint could not be used because it did not behave like a rigid connection, which made Articulation Body necessary. The previous problem with Articulation Body—no movement during deterministic actions—was caused by not converting between radians and degrees. Joint values (jointPosition) in the Articulation Body are read in radians, so without converting them to degrees, the joint angles are nearly indistinguishable. This caused incorrect observations and resulted in no movement. However, after fixing this problem and training with PPO, it was difficult to observe any meaningful increase in the reward.

## Environment
### Unity
- Unity Editor: 6000.3.0f1
- ML Agents: 4.0.2
- Sentis: 2.5.0

### Python
- Python 3.10.12
