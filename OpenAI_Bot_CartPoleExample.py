import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#-------------------------------------------------------------------------------

	#MAIN STRUCTURES AND VARIABLES

#-------------------------------------------------------------------------------
ENV_NAME = 'CartPole-v0'
MODEL_NAME = 'openAI-{}.model'.format(ENV_NAME)
TRAIN_DATA_NAME = 'openAI-{}-trainingdata.npy'.format(ENV_NAME)
LR = 1e-3
gamesToTrain = 100000 #Number os games to play with random actions to gather training data.
stepsGoal = 500 #Goal of steps to take in each game.
scoreMinimum = 70 #Minimum score of a game to be added to the training data.
modelPlayNumber = 100 #Number of games for the model to play after training.

env = gym.make(ENV_NAME) #Creates the game environment.
env.reset()
envActionsPossibleNum = env.action_space.n

#-------------------------------------------------------------------------------

	#FUNCTIONS

#-------------------------------------------------------------------------------
def playGamesWithRandomActions():
	gamesToPlayWithRandom = 5
	for game in range(gamesToPlayWithRandom):
		env.reset()
		for _ in range(200):
			env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done: break
	print('Finished all preliminar games.') #Statistical purposes.

def createInitialPopulation():
	trainingData = []
	totalScores = 0 #Sum of all the games' scores to calculate the average.
	acceptedGames = 0 #Number of games that scored higher than scoreMinimum.
	#Plays all the training games only with possible random actions.
	for _ in range(gamesToTrain):
		env.reset() #Resets the game environment.
		score = 0
		gameMemory = []
		prevObservation = []
		#Tries to reach the goal of steps.
		for _ in range(stepsGoal):
			#Creates a random action from a set of all the possible action.
			action = env.action_space.sample()
			#Adds [obeservation, action] to the game data.
			if len(prevObservation) > 0: gameMemory.append([prevObservation, action])
			'''Executes the action. Returns
			- Observation - the new state of the environment
			- Reward - Points given for the action taken
			- Done - if the game is over
			- Info - other related information
			'''
			prevObservation, reward, done, info = env.step(action)
			score += reward
			if done: break #If the game is over stop taking actions.
		totalScores += score
		#If scoreMinimum is reached, adds [observation, action] to trainingData.
		if score >= scoreMinimum:
			acceptedGames += 1 #Statistical purposes.
			#Transforms the action into a One-Hot vector.
			for data in gameMemory:
				output = [0] * envActionsPossibleNum
				output[data[1]] = 1
				trainingData.append([data[0], output]) #Appends [observation, action] to trainingData
	print('Played {} training games. Averaged {} points. Accepted {} training games'.\
		format(gamesToTrain, totalScores / gamesToTrain, acceptedGames)) #Statistical purposes.
	#Saves the processed training data to save time when retraining (with the same data).
	np.save(TRAIN_DATA_NAME, np.array(trainingData))
	return trainingData

def modelCreate(inputSize):
	network = input_data(shape=[None, inputSize, 1], name='input')
	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)
	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)
	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.8)
	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)
	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)
	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
	return tflearn.DNN(network, tensorboard_dir='log')

def modelFit(model, trainingData):
	#Training Data
	X = np.array([i[0] for i in trainingData]).reshape(-1, len(trainingData[0][0]), 1)
	Y = [i[1] for i in trainingData]
	#Fits the model.
	model.fit( {'input' : X}, {'targets' : Y}, n_epoch = 5, snapshot_step = 500, 
				show_metric = True, run_id = MODEL_NAME)
	model.save(MODEL_NAME) #Saves the model so retraining isn't necessary.
	return model

def modelPlayGame(model):
	env.reset() #Resets the game environment.
	gameMemory = []
	action = env.action_space.sample() #Initial move when there is no game observation.
	prevObservation, reward, done, _ = env.step(action) #Takes the random action.
	score = reward
	for _ in range(stepsGoal):
		if done: break #If the game is over stop taking actions.
		#env.render() #Renders the environment to see what is happening.
		prediction = model.predict(prevObservation.reshape(-1, len(prevObservation), 1))
		action = np.argmax(prediction[0])
		gameMemory.append([prevObservation, action])
		prevObservation, reward, done, _ = env.step(action)
		score += reward
	return score, gameMemory

def modelTrainAndPlay():
	trainingData = []
	print('Processing training data...')
	try:
		trainingData = np.load(TRAIN_DATA_NAME).tolist() #Loads the training data.
		print('Loading saved training data...')
	except IOError:
		print('Creating new training data...')
		trainingData = createInitialPopulation() #Creates the training data.	
	print('Done processing.\n')

	print('Creating model...')
	model = modelCreate(len(trainingData[0][0])) #Creates the model
	print('Done creating model.\n')

	if os.path.exists('{}.meta'.format(MODEL_NAME)): #Loads the model if it exists.
		model.load(MODEL_NAME)
		print('Done loading model.\n')
	else: #Trains the model if it doesn't exist.
		print('Training model...')
		model = modelFit(model, trainingData)
		print('Done training model.\n')

	totalScores = 0
	newTrainingData = []
	#Plays the game modelPlayNumber of times after being trained.
	print('Playing games...')
	for i in range(modelPlayNumber):
		score, gameMemory = modelPlayGame(model)
		totalScores += score
		for data in gameMemory:
			output = [0] * envActionsPossibleNum
			output[data[1]] = 1
			#Appends [observation, action] to trainingData
			newTrainingData.append([data[0], output])
	np.save(TRAIN_DATA_NAME, np.array(newTrainingData))
	print('Played {} games. Averaged {} points.'.\
		format(modelPlayNumber, totalScores / modelPlayNumber))

#-------------------------------------------------------------------------------

	#CODE EXECUTION

#-------------------------------------------------------------------------------
playGamesWithRandomActions()
#modelTrainAndPlay()
