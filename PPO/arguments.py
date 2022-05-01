"""
	This file contains the arguments to parse at command line.
	File main.py will call get_args, which then the arguments
	will be returned.
"""
import argparse


def get_args():
	"""
		Description:
		Parses arguments at command line.
		Parameters:
			No parameters
		Return:
			args - the arguments parsed
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'

	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')

	parser.add_argument('--rival_actor_model', dest='rival_actor_model', type=str, default='PPO/weights/rival_actor.pth')

	args = parser.parse_args()

	return args
