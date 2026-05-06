# Under-Construction
Making an agent walk with legs using reinforcement learning in Unity was a real challenge for me. After several attempts, it finally worked with two main factors: Rad/Deg conversion and Evolution Strategies. Configurable Joint could not be used because it did not behave like a rigid connection, which made Articulation Body necessary. The previous problem with Articulation Body—no movement during deterministic actions—was caused by not converting between radians and degrees.

## Environment
### Unity
- Unity Editor: 6000.3.0f1
- ML Agents: 4.0.2
- Sentis: 2.5.0

### Python
- Python 3.10.12
