# Pitch Prediction using Deep Learning: Applying AI to the sport of baseball

This project implements an ensemble of neural networks to predict pitch type and location in baseball. More specifically, LSTMs, GRUs, LSTMs, Attention-based LSTMs Leveraging the Statcast dataset, the system processes sequential pitch data to forecast key characteristics of the next pitch, including its type (e.g., fastball, curveball) and location within the strike zone.

The core innovation of this project lies in its comprehensive approach to pitch prediction. While previous studies have often focused solely on pitch type or, to a lesser extent, location, this project aims to predict *both* aspects of a pitch simultaneously. By employing a multi-task learning framework, the model is trained to optimize for both pitch type and location, thereby enhancing its predictive accuracy and utility for coaches, players, and analysts.

This project was developed as the final project to complete the BSc Computer Science program at the University of London. It is based on the Machine Learning on a Public Dataset template.

## Features

- **Multi-task Prediction**: Simultaneously predicts pitch type and location. Most existing models focus only on pitch type.
- **Statcast Dataset Integration**: Utilizes MLB's Statcast data for training and evaluation. This makes the model easily adaptable to real-world scenarios, as it is trained on actual game data.
- **Sequential Neural Networks**: Utilizes advanced architectures like LSTMs, GRUs, and Transformers to model the sequential nature of baseball pitches. By understanding the context of prior pitches within an at-bat, the model effectively captures temporal dependencies, improving prediction performance.

## Repository Structure

```
pitch_prediction_using_ML/
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks used for exploration and analysis
├── utils/              # Utility functions for data processing and model evaluation
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
```

## Acknowledgments

- MLB's Statcast for providing the dataset.
- Open-source libraries such as TensorFlow, PyTorch, and Matplotlib.
- The baseball analytics community for inspiring this project.

## Related Work

This project was inspired by prior research in the field of pitch prediction. Notably:

- Lee's work on predicting pitch type and location using ensemble models of deep neural networks provided valuable insights into combining multiple models for predicting both type and location. [Jae Sik Lee. 2022. Prediction of pitch type and location in baseball using ensemble model of deep neural networks. JSA 8, 2 (July 2022), 115–126. https://doi.org/10.3233/JSA-200559]
- Yu et al. study on attention-based LSTMs for pitch prediction highlighted the potential of attention mechanisms in capturing temporal dependencies in sequential data. [Chih-Chang Yu, Chih-Ching Chang, and Hsu-Yung Cheng. 2022. Decide the Next Pitch: A Pitch Prediction Model Using Attention-Based LSTM. In 2022 IEEE International Conference on Multimedia and Expo Workshops (ICMEW), July 18, 2022. IEEE, Taipei City, Taiwan, 1–4. https://doi.org/10.1109/ICMEW56448.2022.9859411]

These studies helped shape the approach and methodology used in this project.

