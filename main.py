from Room import Room
import dqn
import time

token = "thr1.AAAAAFwG5wJArFEqqbgBRg.W5GvVGGkhM8"  # from : https://www.haxball.com/headlesstoken
room = Room("room", token, 2)
room.reset()
#dqn.train_P1(room, 100000)
time.sleep(1000)