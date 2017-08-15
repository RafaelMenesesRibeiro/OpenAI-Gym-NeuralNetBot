# OpenAI-Gym-NeuralNetBot

Deep Neural Network that learns to play OpenAI gym's games
Uses Tensorflow library for the DNN model.

* Starts by playing the game 100000 times (gamesToTrain variable) only with possible random moves.
* Adds the game data ([observation, action]) from games that finished with a score over 100 (scoreMinimum) to the training data.
* Trains with that data.
* Plays the game 100 times (modelPlayNumber) with the DNN's prediction (with the current observation as input).
* Adds the new game data to the training data.
* Saves it so the next time it runs it can learn from more and better data.

## Requirements
* Gym
* Tensorflow / Tensorflow-gpu
  - CUDA Toolkit 8.0 (optional - to use with the gpu version)
  - cuDNN 7 (optional - to use with the gpu version)
* TFlearn
* Numpy

## How To Use
```python
#Set ENV_NAME in file.py:15 to the name of environment to learn
ENV_NAME = 'CartPole-v1'
```

```bash
#Run file.py to train/load the model.
$ python file.py
```
