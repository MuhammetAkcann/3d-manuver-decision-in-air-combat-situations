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

    parser.add_argument('--mode', dest='mode', type=str, default='train')  # can be 'train' or 'test'

    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')

    parser.add_argument('--rival_actor', dest='rival_actor', type=str, default='PPO/weights/backup/random_rival/ppo_actor.pth')

    parser.add_argument('--shadows', dest='shadows', type=str, nargs='+', default="")
    parser.add_argument('--noises', dest='noises', type=str, nargs='+', default="")

    args = parser.parse_args()
    shadows = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    noises = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for shadow in args.shadows:
        if shadow == 'pos_dif_x':
            shadows[0] = 0
        elif shadow == 'pos_dif_y':
            shadows[1] = 0
        elif shadow == 'pos_dif_z':
            shadows[2] = 0
        elif shadow == 'roll':
            shadows[3] = 0
        elif shadow == 'pitch':
            shadows[4] = 0
        elif shadow == 'yaw':
            shadows[5] = 0
        elif shadow == 'speed':
            shadows[6] = 0
        elif shadow == 'rival_roll':
            shadows[7] = 0
        elif shadow == 'rival_pitch':
            shadows[8] = 0
        elif shadow == 'rival_yaw':
            shadows[9] = 0
        elif shadow == 'rival_speed':
            shadows[10] = 0

    for noise in args.noises:
        if noise == 'pos_dif_x':
            noises[0] = 1
        elif noise == 'pos_dif_y':
            noises[1] = 1
        elif noise == 'pos_dif_z':
            noises[2] = 1
        elif noise == 'roll':
            noises[3] = 1
        elif noise == 'pitch':
            noises[4] = 1
        elif noise == 'yaw':
            noises[5] = 1
        elif noise == 'speed':
            noises[6] = 1
        elif noise == 'rival_roll':
            noises[7] = 1
        elif noise == 'rival_pitch':
            noises[8] = 1
        elif noise == 'rival_yaw':
            noises[9] = 1
        elif noise == 'rival_speed':
            noises[10] = 1

    args.shadows = shadows
    args.noises = noises
    return args
