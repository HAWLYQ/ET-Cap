import numpy as np
import math

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


def get_new_view_vector(start_view_vec, start_view_x_angle, yaw_action, yaw_rotate, pitch_action, pitch_rotate):
    """
    old_view: previous view vector (x, y, z)
    start_view_x_angle: view angle with (1,0,0), always counterclockwise direction
    yaw_rotate: yaw rotate angle (>0)
    pitch_rotate: pitch rotate angle (>0)
    """
    # calculate new yam angle
    start_view_vec_xy = np.array(list(start_view_vec[0:2])+[0])
    # start_view_x_angle = vector_angle(start_view_vec_xy, [1,0,0])
    
    if yaw_action == 'left':
        end_view_x_angle = (start_view_x_angle + yaw_rotate) % 360
    elif yaw_action == 'right':
        end_view_x_angle = (start_view_x_angle - yaw_rotate) % 360
    else:
        assert yaw_action == 'none'
        end_view_x_angle = start_view_x_angle

    # calculate new pitch angle
    start_view_z_angle = vector_angle(start_view_vec, [0,0,1])

    ## allow z_angle == 0 or 180 
    if pitch_action == 'up':
        end_view_z_angle = min(max(start_view_z_angle - pitch_rotate, 0), 180) # 0.01<=z_angle<=179.99
    elif pitch_action == 'down':
        end_view_z_angle = min(max(start_view_z_angle + pitch_rotate, 0), 180)
    else:
        assert pitch_action == 'none'
        end_view_z_angle = start_view_z_angle
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


def lookat_transfer(position, look_at):
    # print(position, look_at)
    x1 = position[0]
    y1 = position[1]
    z1 = position[2]
    x2 = look_at[0]
    y2 = look_at[1]
    z2 = look_at[2]
    # (x-x1)/(x1-x2) = (y-y1)/(y1-y2) = (z-z1)/(z1-z2)
    if x1 != x2 and x1!=0: # if x1==0, look_at will be as the same as position
        x = 0
        y = (y1-y2)*(x-x1)/(x1-x2) + y1 
        z = (z1-z2)*(x-x1)/(x1-x2) + z1
    elif y1 != y2 and y1!=0: # if y1==0, look_at will be as the same as position
        y = 0
        x = (x1-x2)*(y-y1)/(y1-y2) + x1 # x==x1==x2
        z = (z1-z2)*(y-y1)/(y1-y2) + z1
    else:
        assert z1!=z2 and z1!=0 # if z1==0, look_at will be as the same as position
        z = 0
        x = (x1-x2)*(z-z1)/(z1-z2) + x1 # x==x1==x2
        y = (y1-y2)*(z-z1)/(z1-z2) + y1 # y==y1==y2
    return np.array([x,y,z])

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


def new_pos_and_lookat(old_position, old_lookat, old_view_x_angle, move_action, move_step, yaw_action, yaw_angle, pitch_action, pitch_angle):
    """
    old_position: ndarray
    old_lookat: ndarry
    old_view_x_angle: float
    move_action, yaw_action, pitch_action: str
    move_step, yaw_angle, pitch_angle: float
    """
    new_position = get_new_position_according_rel_actions(old_position, old_view_x_angle, move_action, move_step)
    old_view_vector = old_lookat - old_position
    new_view_vector, new_view_x_angle = get_new_view_vector(old_view_vector, old_view_x_angle, yaw_action, yaw_angle, pitch_action, pitch_angle)
    new_lookat = new_view_vector + new_position
    return new_position, new_lookat, new_view_x_angle, new_view_vector

if __name__ == '__main__':
    # vector_angle([0,0,1], [1, 0, 1])
    pitch_action, yaw_action, end_view_x_angle = rotate_action(start_coor=[0,0,-1], start_view_x_angle=45, end_coor=[0,0,-2])