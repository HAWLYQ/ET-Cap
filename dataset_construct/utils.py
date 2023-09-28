import numpy as np
import math
from icecream import ic

def vector_angle(x, y):
    x=np.array(x)
    y=np.array(y)

    # |x| and |y|
    l_x=np.sqrt(x.dot(x))
    l_y=np.sqrt(y.dot(y))
    if l_x == 0 or l_y == 0:
        return 'nan'

    # x·y
    dot_product=x.dot(y)

    # get cosine：
    cos_=dot_product/(l_x*l_y)

    # get angle (pi)：
    angle=np.arccos(cos_)
    angle = angle*180/np.pi
    # print(x, y, angle)
    return angle


def rotate_action(start_coor, start_view_x_angle, end_coor):
    """
    for start_view_x_angle: 
    if start_coor is not [0,0,x], it's calculated by vector_angle(start_view_vec_xy, [1,0,0])
    if start_coor is [0,0,x] and it's not the first point, it's inherited from last view
    if start_coor is [0,0,x] and it's the first point, it's set as 0
    """
    start_coor = np.array(start_coor)
    end_coor = np.array(end_coor)
    # target_coor = np.array([0,0,0])
    target_coor = np.array([0,0,-0.25]) # target_coor * 0.4 + [0,0,0.1] = [0,0,0]
    start_view_vec = target_coor-start_coor
    # start_view_vec_xy = np.array(list(start_view_vec[0:2])+[0])
    start_view_z_angle = vector_angle(start_view_vec, [0,0,1])

    
    ## 0<=z_angle<=180, x_angle inherit previous x_angle
    start_view_z_angle = min(max(start_view_z_angle, 0), 180)
    
    # print('angle between start view and z+: ', start_view_z_angle)
    # print('angle between start view(xy) and x+: ', start_view_x_angle)

    end_view_vec = target_coor-end_coor
    end_view_vec_xy = np.array(list(end_view_vec[0:2])+[0])
    end_view_z_angle = vector_angle(end_view_vec, [0,0,1])

    # 0<=z_angle<=180
    end_view_z_angle = min(max(end_view_z_angle, 0), 180)
    end_view_x_angle = vector_angle(end_view_vec_xy, [1,0,0])

    if end_view_x_angle == 'nan':
        end_view_x_angle = start_view_x_angle
    else:
        if end_view_vec[1] < 0:
            assert end_view_x_angle < 180
            end_view_x_angle = 360-end_view_x_angle

    # print('angle between end view and z+: ', end_view_z_angle) 
    # print('angle between end view(xy) and x+: ', end_view_x_angle)
    pitch_action = angle_to_pitch_rotate_actions(start_view_z_angle, end_view_z_angle)
    yaw_action = angle_to_yaw_rotate_action(start_view_x_angle, end_view_x_angle)
    # print('pitch action: ', pitch_action)
    # print('yaw action:', yaw_action)
    # print('end x angle:', end_view_x_angle)
    return pitch_action, yaw_action, end_view_x_angle


def angle_to_pitch_rotate_actions(start_z_angle, end_z_angle):
    """
    start_z_angle: the angle between start view and z+ 0~180
    end_z_angle: the angle between end view and z+ 0~180
    """
    rotate_angle = end_z_angle-start_z_angle
    if rotate_angle > 0:
        # action = 'down'+' '+str(rotate_angle)
        action = {'pitch': ['down', round(rotate_angle, 2)]}
    elif rotate_angle < 0:
        # action = 'up'+' '+str(-rotate_angle)
        action = {'pitch': ['up', round(-rotate_angle, 2)]}
    else:
        # action = 'down/up'+' '+str(rotate_angle)
        action = {'pitch':['none', 0]}
    return action

def angle_to_yaw_rotate_action(start_view_x_angle, end_view_x_angle):
    """
    start_view_x_angle: the angle between x+ and start_view_xy 0~360
    end_view_x_angle: the angle between x+ and end_view_xy 0~360
    """
    # print(start_view_x_angle, type(start_view_x_angle))
    # print(end_view_x_angle, type(end_view_x_angle))
    rotate_angle = end_view_x_angle - start_view_x_angle
    if rotate_angle == 0:
        # action = 'left/right'+' '+str(0)
        action = {'yaw': ['none', 0]}
    elif rotate_angle > 0 and rotate_angle <= 180:
        # action = 'left'+' '+str(rotate_angle)
        action = {'yaw': ['left', round(rotate_angle, 2)]}
    elif rotate_angle > 0 and rotate_angle > 180:
        # action = 'left'+' '+str(rotate_angle)
        action = {'yaw': ['right', round(360-rotate_angle, 2)]}
    elif rotate_angle < 0 and -rotate_angle <= 180:
        # action = 'right'+' '+str(-rotate_angle)
        action = {'yaw': ['right', round(-rotate_angle, 2)]}
    elif rotate_angle < 0 and -rotate_angle > 180:
        # action = 'right'+' '+str(-rotate_angle)
        action = {'yaw': ['left', round(360+rotate_angle, 2)]}
    return action


def relative_move_action(start_coor, end_coor, start_view_x_angle):
    move_vec = np.array(end_coor) - np.array(start_coor)
    action  = {'move':{'fb':[], 'rl':[], 'ud':[]}}
    # up/down move action
    if move_vec[2] > 0:
        # action = 'move up'+''+str(1)
        # action = {'move':'up'}
        action['move']['ud']+=['up', int(move_vec[2])]
    if move_vec[2] < 0:
        # action = 'move down'+''+str(1)
        # action = {'move':'down'}
        action['move']['ud']+=['down', int(-move_vec[2])]
    
    # left/right/forward/backward move action
    rel_x, rel_y = xy_transfer(end_coor[0], end_coor[1], start_coor[0], start_coor[1], start_view_x_angle)
    rel_x = round(rel_x)
    rel_y = round(rel_y)
    if rel_x > 0:
        action['move']['fb']+=['forward', rel_x]
    if rel_x < 0:
        action['move']['fb']+=['backward', -rel_x]
    if rel_y > 0:
        action['move']['rl']+=['left', rel_y]
    if rel_y < 0:
        action['move']['rl']+=['right', -rel_y]
    
    for k,v in action['move'].items():
        if len(v) == 0:
            v+=['none', 0]
    return action

def xy_transfer(raw_x, raw_y, origin_raw_x, origin_raw_y, counterclockwise_angle):
    # 
    mid_x = raw_x-origin_raw_x
    mid_y = raw_y-origin_raw_y
    # counterclockwise_angle means counterclockwise_angle from raw x+ to new x+
    new_x = mid_x * math.cos(math.pi/180 * counterclockwise_angle) + mid_y * math.sin(math.pi/180 * counterclockwise_angle)
    new_y = mid_y * math.cos(math.pi/180 * counterclockwise_angle) - mid_x * math.sin(math.pi/180 * counterclockwise_angle)
    # new_x -= origin_raw_x
    # new_y -= origin_raw_y
    # ic(raw_x, raw_y, origin_raw_x, origin_raw_y, counterclockwise_angle)
    # ic(new_x, new_y)

    return new_x, new_y


def get_new_view_vector(start_view_vec, start_view_x_angle, yaw_action, yaw_rotate, pitch_action, pitch_rotate):
    """
    old_view: previous view vector (x, y, z)
    yaw_rotate: yaw rotate angle (>0:left, <0:right)
    pitch_rotate: pitch rotate angle (>0:down, <0:up)
    """
    # calculate new yaw angle
    start_view_vec_xy = np.array(list(start_view_vec[0:2])+[0])
    # start_view_x_angle = vector_angle(start_view_vec_xy, [1,0,0])

    """if start_view_vec[1] < 0:
        start_view_x_angle = 360-start_view_x_angle"""
    
    print('start view x angle:', start_view_x_angle)
    if yaw_action == 'left':
        end_view_x_angle = (start_view_x_angle + yaw_rotate ) % 360
    else:
        end_view_x_angle = (start_view_x_angle - yaw_rotate) % 360
    
    print('end view x angle:', end_view_x_angle)

    # calculate new pitch angle
    print('start view vector:', start_view_vec)
    start_view_z_angle = vector_angle(start_view_vec, [0,0,1])

    print('start view z angle:', start_view_z_angle)
    ## scheme 2: allow z_angle == 0 or 180 
    if pitch_action == 'up':
        end_view_z_angle = min(max(start_view_z_angle - pitch_rotate, 0), 180) # 0.01<=z_angle<=179.99
    else:
        end_view_z_angle = min(max(start_view_z_angle + pitch_rotate, 0), 180)
    print('end view z angle:', end_view_z_angle)
    # calculate new x,y,z of a unit vector
    new_z = 1 * math.cos(math.pi/180 * end_view_z_angle)

    if end_view_z_angle != 0 and end_view_z_angle != 180:
        new_xy = 1 * math.sin(math.pi/180 * end_view_z_angle)
        new_x = new_xy * math.cos(math.pi/180 * end_view_x_angle)
        new_y = new_xy * math.sin(math.pi/180 * end_view_x_angle)
    else:
        new_y = 0
        new_x = 0
    return np.array([new_x, new_y, new_z]), end_view_x_angle


def get_new_position_according_rel_actions(old_position, start_view_x_angle, move_actions, move_steps):
    """
    old_position: (x,y,z)
    start_view_x_angle: float
    move_actions: list of relative action
    move_steps: list of steps
    """
    assert len(move_actions) == len(move_steps)
    rel_x = 0
    rel_y = 0
    new_z = old_position[2]
    # calculate position in camera xy axis after move
    for i in range(len(move_actions)):
        move_action = move_actions[i]
        move_step = move_steps[i]
        if move_action == 'up':
            new_z = min(new_z + move_step, 10)
        if move_action == 'down':
            new_z = max(new_z - move_step, 0)
        if move_action == 'forward':
            rel_x = move_step
        if move_action == 'backward':
            rel_x = -move_step
        if move_action == 'left':
            rel_y = move_step
        if move_action == 'right':
            rel_y = -move_step
    
    new_x, new_y = xy_transfer(rel_x, rel_y, np.sqrt(old_position[0]*old_position[0]+old_position[1]*old_position[1]), 0, 360-start_view_x_angle)
    new_x =  min(max(round(new_x), -10), 10)
    new_y =  min(max(round(new_y), -10), 10)
    new_position = np.array([new_x, new_y, new_z])
    
    return new_position


def lookat_transfer(position, look_at):
    # print(position, look_at)
    x1 = position[0]
    y1 = position[1]
    z1 = position[2]
    x2 = look_at[0]
    y2 = look_at[1]
    z2 = look_at[2]
    # (x-x1)/(x1-x2) = (y-y1)/(y1-y2) = (z-z1)/(z1-z2)
    if x1 != x2:
        x = 0
        y = (y1-y2)*(x-x1)/(x1-x2) + y1 
        z = (z1-z2)*(x-x1)/(x1-x2) + z1
    elif y1 != y2:
        y = 0
        x = (x1-x2)*(y-y1)/(y1-y2) + x1 # x==x1==x2
        z = (z1-z2)*(y-y1)/(y1-y2) + z1
    else:
        assert z1!=z2
        z = 0
        x = (x1-x2)*(z-z1)/(z1-z2) + x1 # x==x1==x2
        y = (y1-y2)*(z-z1)/(z1-z2) + y1 # y==y1==y2
    return np.array([x,y,z])


def new_pos_and_lookat(old_position, old_lookat, old_view_x_angle, move_action, move_step, yaw_action, yaw_angle, pitch_action, pitch_angle):
    """
    old_position: ndarray
    old_lookat: ndarry
    old_view_x_angle: float
    move_action: list of str
    yaw_action, pitch_action: str
    move_step: list of float
    yaw_angle, pitch_angle: float
    """
    # print(old_position, old_lookat)
    new_position = get_new_position_according_rel_actions(old_position, old_view_x_angle, move_action, move_step)
    
    # print(old_position, old_lookat)
    old_view_vector = old_lookat - old_position
    # print('old view vector:', old_view_vector)
    new_view_vector, new_view_x_angle = get_new_view_vector(old_view_vector, old_view_x_angle, yaw_action, yaw_angle, pitch_action, pitch_angle)
    new_lookat = new_view_vector + new_position
    new_lookat = lookat_transfer(new_position, new_lookat) # set x in new_lookat as 0
    return new_position, new_lookat, new_view_x_angle


def move_action(start_coor, end_coor):
    move_vec = np.array(end_coor) - np.array(start_coor)
    # assert there is at most one variation amone x,y,z
    assert len(np.nonzero(move_vec)[0]) <= 1 
    # assert sum(move_vec) <= 1
    if move_vec[0] > 0:
        # action = 'move backward'+''+str(1)
        # action = {'move':'backward'}
        action = {'move':['backward', int(move_vec[0])]}
    elif move_vec[0] < 0:
        # action = 'move forward'+''+str(1)
        # action = {'move':'forward'}
        action = {'move':['forward', int(-move_vec[0])]}
    elif move_vec[1] > 0:
        # action = 'move right'+''+str(1)
        # action = {'move':'right'}
        action = {'move':['right', int(move_vec[1])]}
    elif move_vec[1] < 0:
        # action = 'move left'+''+str(1)
        # action = {'move':'left'}
        action = {'move':['left', int(-move_vec[1])]}
    elif move_vec[2] > 0:
        # action = 'move up'+''+str(1)
        # action = {'move':'up'}
        action = {'move':['up', int(move_vec[2])]}
    elif move_vec[2] < 0:
        # action = 'move down'+''+str(1)
        # action = {'move':'down'}
        action = {'move':['down', int(-move_vec[2])]}
    else:
        # action = 'no move'
        action = {'move':['none', 0]}
    return action



def test_action_generation():

    start_coor = np.array([1,0,0])
    # end_coor = np.array([1,0,1])
    end_coor = np.array([1,2,1])


    # start_coor = np.array([10,-9,5])
    # end_coor = np.array([10,-5,5])
    
    # start_coor = np.array([1, 1, 0])
    # end_coor = np.array([1,1,0])

    target_coor = np.array([0,0,-0.25])
    start_view_vec = target_coor-start_coor
    start_view_vec_xy = np.array(list(start_view_vec[0:2])+[0])
    start_view_x_angle = vector_angle(start_view_vec_xy, [1,0,0])
    if start_view_x_angle == 'nan':
        start_view_x_angle = 0
    else:
        if start_view_vec[1] < 0:
            assert start_view_x_angle < 180
            start_view_x_angle = 360-start_view_x_angle

    move_act = relative_move_action(start_coor, end_coor, start_view_x_angle)
    pitch_act, yaw_act, end_view_x_angle = rotate_action(start_coor, start_view_x_angle, end_coor)

    print('move action:', move_act)
    print('pitch action:', pitch_act)
    print('yaw action:', yaw_act)

    print('===========reduction==========')
    print('start coor', start_coor)
    """reduct_end_coor, reduct_lookat, reduct_view_x_angle = new_pos_and_lookat(start_coor, target_coor, start_view_x_angle, 
                                                    move_act['move'][0], move_act['move'][1],
                                                    yaw_act['yaw'][0], yaw_act['yaw'][1],
                                                    pitch_act['pitch'][0], pitch_act['pitch'][1])"""
    move_act = move_act['move']
    move_actions = [v[0] for k, v in move_act.items()]
    move_steps = [v[1] for k, v in move_act.items()]
    reduct_end_coor, reduct_lookat, reduct_view_x_angle = new_pos_and_lookat(start_coor, target_coor, start_view_x_angle, 
                                                    move_actions, move_steps,
                                                    yaw_act['yaw'][0], yaw_act['yaw'][1],
                                                    pitch_act['pitch'][0], pitch_act['pitch'][1])

    print('reduct end coor:', reduct_end_coor)
    print('reduct end lootat:', [round(x, 4) for x in reduct_lookat.tolist()])


if __name__ == '__main__':
    # vector_angle([0,0,1], [1, 0, 1])
    # pitch_action, yaw_action, end_view_x_angle = rotate_action(start_coor=[0,0,-1], start_view_x_angle=45, end_coor=[0,0,-2])
    test_action_generation()
    # xy_transfer(raw_x=0, raw_y=-5, origin_raw_x=0, origin_raw_y=0, counterclockwise_angle=0)