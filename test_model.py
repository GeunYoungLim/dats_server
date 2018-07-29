# test_model.py

import numpy as np
#from grabscreen import grab_screen
import cv2
import time
#from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet
#from getkeys import key_check
from dats.core.server import remote_model
import random

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

t_time = 0.09


model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

#    last_time = time.time()
#    for i in list(range(4))[::-1]:
#        print(i+1)
#        time.sleep(1)

paused = False
def processor(server, stream_frame, recv_data):
	if not paused:
		# 800x600 windowed mode
		#screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
		#screen = grab_screen(region=(0,40,800,640))
		#print('loop took {} seconds'.format(time.time()-last_time))
		print('shape'+str(stream_frame.shape))
		#last_time = time.time()
		screen = cv2.cvtColor(stream_frame, cv2.COLOR_RGB2GRAY)
		screen = cv2.resize(screen, (160,120))

		prediction = model.predict([screen.reshape(160,120,1)])[0]
		print(prediction)

		turn_thresh =0.75
		fwd_thresh = 0.70

		if prediction[1] > fwd_thresh:
			return bytearray([0, 1, 0])
		elif prediction[0] > turn_thresh:
			return bytearray([1, 0, 0])
		elif prediction[2] > turn_thresh:
			return bytearray([0, 0, 1])
		else:
			return bytearray([0, 1, 0])
			
		
       # keys = key_check()

        # p pauses game and can get annoying.
'''
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)
'''
s1 = remote_model(addr='0.0.0.0', stream_port = 8888, command_port=7777)
s1.set_loop_processor(processor)
s1.open()
