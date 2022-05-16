#!/usr/bin/env python3
import glob
import os
import sys
import random
import time
import sys
import numpy as np
import cv2
import math
from collections import deque
import tensorflow as tf
# from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard
import tensorflow.keras.backend as backend
from threading import Thread
from tqdm import tqdm


sys.path.append('')
import carla


class CarEnv:

    def __init__(self):

    def reset(self):

    def collision_data(self, event):

    def process_img(self, image):

    def step(self, action):


class DQNAgent:
    def __init__(self):

    def create_model(self):

    def update_replay_memory(self, transition):

    def train(self):

    def get_qs(self, state):

    def train_in_loop(self):


class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):

    def set_model(self, model):

    def on_epoch_end(self, epoch, logs=None):

    def on_batch_end(self, batch, logs=None):

    def on_train_end(self, _):

    def _write_logs(self, logs, index):

    def update_stats(self, **stats):
      


if __name__ == '__main__':
