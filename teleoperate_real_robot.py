from robot import Robot
from dynamixel import Dynamixel
import time
import pickle

import cv2

camera = cv2.VideoCapture(2)

leader_dynamixel = Dynamixel.Config(baudrate=57600, device_name='/dev/ttyUSB0').instantiate()
follower_dynamixel = Dynamixel.Config(baudrate=57600, device_name='/dev/ttyUSB1').instantiate()
follower = Robot(follower_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
leader = Robot(leader_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
follower.set_trigger_torque()
leader.set_trigger_torque()

save = True

follower_pos_init = follower.read_position()
leader_pos_init = leader.read_position()

# Set camera resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def main(idx=0):
    global leader_dynamixel, follower_dynamixel, follower, leader, save

    if input('start > ').strip()[0] not in ['y', 'Y']:
        exit(0)

    data = []

    while True:
        follower_pos = follower.read_position()
        leader_pos = leader.read_position()
        print(f'follower: {follower_pos}, leader: {leader_pos}')
        ff_set = [k + j - i for i, j, k in zip(leader_pos_init, leader_pos, follower_pos_init)]
        follower_pos_offset = [j - i for i, j in zip(leader_pos_init, leader_pos)]
        follower.set_goal_pos(ff_set)
        # time.sleep(0.1)
        ret, frame = camera.read()
        if not ret:
            continue
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # camera.release()
            cv2.destroyAllWindows()
            # del leader_dynamixel, follower_dynamixel, follower, leader
            break
        if save:
            data.append(
                dict(
                    follower_pos=follower_pos_offset,
                    frame=frame
                )
            )

    if save:
        with open(f'output/data-{idx}.pkl', 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    for idx in range(48, 60):
        main(idx=idx)
        follower.set_goal_pos(follower_pos_init)
        # leader.set_goal_pos(leader_pos_init)
