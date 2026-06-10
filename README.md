## Liquid Neural Network Audio Deepfake Detector

This project investigates **audio deepfake detection** using **Liquid Neural Networks (LNNs)** to model temporal speech dynamics in continuous time. 
Unlike conventional RNN/LSTM-based detectors, the proposed approach leverages **Liquid Time-Constant (LTC) neurons** governed by learnable differential equations, enabling improved generalization a[...]
The model is evaluated on **ASVspoof**, **MLAAD**, and **In-the-Wild (ITW)** audio deepfake benchmarks, focusing on cross-domain robustness.

---

## 📄 Research Paper

For detailed methodology and experimental results, refer to the research paper:
[**Liquid Neural Networks for Audio Deepfake Detection**](https://drive.google.com/file/d/1hRLsjtD7B9ZBexY1Z4OILbIpQeulUB7E/view?usp=drive_link)

---

## Model Architecture


<img src="assets/Architechture_2.jpg" width="800">


The pipeline converts raw audio into **mel-spectrograms**, extracts local representations using a convolutional front-end, and models temporal dependencies using an **LTC-based ODE cell**, followe[...]

---

## Liquid Neural Network Dynamics

The hidden state of the Liquid Neural Network evolves according to a continuous-time differential equation:

dx(t)/dt = -(1/τ) x(t) + σ(W x(t) + U u(t) + b)


where:
- \(x(t)\) represents the neuron state,
- \(u(t)\) is the input feature vector,
- \(\tau\) is a learnable time constant,
- \(W, U, b\) are trainable parameters,
- \(\sigma(\cdot)\) denotes a nonlinear activation.

The ODE is numerically solved using a Runge–Kutta (RK4) scheme, enabling adaptive temporal modeling for variable-length and in-the-wild audio signals.
