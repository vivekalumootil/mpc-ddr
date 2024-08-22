import numpy as np
import math
from controller import Robot
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint

def robot_tf(x, y, rob):
    x = x - rob[0]
    y = y - rob[1]
    inverse_rotation = np.array([[math.cos(-rob[2]), -math.sin(-rob[2])], [math.sin(-rob[2]), math.cos(-rob[2])]])
    p = np.dot(inverse_rotation, np.array([x, y]))
    r_x = p[0]
    r_y = p[1]
    return r_x, r_y

def line_from_points(P1, P2):
    x1, y1 = P1
    x2, y2 = P2
    return np.array([y1-y2, x2-x1, x1*(y1+y2)-y1*(x1+x2)])

iterations = 20 # number of timesteps completed
hor = 30 # mpc horizon length
ts = 0.2 # timestep (lower results in more accurate mechanics)

init_pose = np.array([0, 0, np.pi/2])
# width 5, radius 1, timestep 0.2, horizon length 5
robot = Robot(ts, hor, init_pose)
x = np.array([-10, 1, 3])

x_plt = np.zeros(iterations)
y_plt = np.zeros(iterations)
d_plt = np.zeros(iterations) 

a = np.array([0, -3, 1 , 7]) # target line is (0, 30) <-> (1, 130)
x_plot_1 = np.linspace(-10, 20, 100)
y_plot_1 = 10*x_plot_1-3

for i in range(iterations):
    print(f'Iteration {i}')
    # transforming target line into robot frame
    goal = line_from_points(robot_tf(a[0], a[1], robot.world), robot_tf(a[2], a[3], robot.world))
    opt = robot.optimize(np.array([0, 0, 0]), goal)
    opt_sol = opt.x.reshape((hor, 2))
    print(f'Input: {opt_sol[0]}')
    print(f'Robot position: {robot.world}')
    x_plt[i] = robot.world[0]
    y_plt[i] = robot.world[1]
    d_plt[i] = robot.world[2]
    # move robot 
    robot.move(np.array([opt_sol[0][0], opt_sol[0][1]]))

# plot
plt.quiver(x_plt, y_plt, np.cos(d_plt), np.sin(d_plt), color='mediumaquamarine', width=0.005)
plt.axis('equal')
plt.plot(x_plot_1, y_plot_1)
plt.show()
