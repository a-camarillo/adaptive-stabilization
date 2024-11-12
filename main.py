import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

state1 = np.array([],dtype=np.complex128)
state2 = np.array([],dtype=np.complex128)
system_input = np.array([],dtype=np.complex128)

# The system for the simulation is the following second-order nonlinear system
# x1_dot = h1(t)*x2 + f1(x1)
# x2_dot = h2(t)*u + f2(x)
# where the signs of h1(t) and h2(t) are unknown and x = [x1 x2] transposed

# The system represents a two-stage chemical reactor with Delay-Free Recycle Streams
# and the functions from the mass-balance equations are
# h1(t) = (1-R2)/V1
# f1(x1)=-((1/theta1+k1)*x1
# h2(t) = F2/V2
# f2(x) = (R1/V2)*(x1)^2 - ((1/theta2) + k2)*x2

# Where the recylce flow rates Ri are
R1 = R2 = 0.55

# The reactor volumes Vi are
V1 = V2 = 0.3

# The feed rate F2 is
F2 = 0.3

# and the reactor residence times theta_i and reaction constants ki are
theta1 = theta2 = 1
k1 = k2 = 1

#known functions
small_gamma1 = h1_bar = 2.5
h2_bar = 2
# the second known function gamma2 is given as a function of x
def gamma2(x1):
    return (x1**2)+2

design_parameter =0.01

# with initial conditions
x_init = np.array([1, -1.5])

def beta1(x1):
    # ~p1(x1) is given as gamma_1(x1)*|ksi_1|^(2/5)
    p1 = 2.5*(np.abs((x1))**(2/5))
    return 2 + p1
    

def smoothing_function(tuning_parameter):
    # NOTE:
    # not sure if this is actually a smoothing function as the paper isn't
    # clear but this function is increasing with respect to tuning parameter k

    # the paper gives H(k) = 0.37(k+1)
    return 0.37*(tuning_parameter+1)

def switching_rule(tuning_parameter):
    # for a second order system the combination of signs exists for 2^2
    # cases and the switching vector can be defined below
    switching_vector = np.array([
        [-1, 1],
        [1, -1],
        [1, 1],
        [-1, -1]
    ], dtype=np.complex128)

    match tuning_parameter:
        case 0:
            # when k=0, take the first column of the switching vector
            return smoothing_function(tuning_parameter)*switching_vector[0]
        case 1:
            # when k=1, take the second column of the switching vector
            return smoothing_function(tuning_parameter)*switching_vector[1]
        case 2:
            # when k=2, take the third column of the switching vector
             return smoothing_function(tuning_parameter)*switching_vector[2]
        case _:
            # when k=3, take the last column of the switching vector
            return smoothing_function(tuning_parameter)*switching_vector[3]

# defining some variables/functions for the beta2(x) function
def small_gamma2(x1):
    return (x1**2)+2
p2_hat = 163.7013
def p2_tilde(x1, x2, tuning_parameter):
    # there's no real reason to breaking this up into parts other than
    # trying to make it look nicer, code should not be 190 columns
    part1 = (7/8)*((3/4)**(1/7))*((((np.abs(x1)**(4/5))*(small_gamma2(x1)))**(8/7))+(smoothing_function(tuning_parameter)*(np.abs(x1)**(2/5))*small_gamma2(x1)*(beta1(x1)))**(8/7))
    print(f'tilde part1: {part1}')
    part2 = (np.abs(((x2**(5/3))-(-(smoothing_function(tuning_parameter)*switching_rule(tuning_parameter)[0]*(beta1(x1)))**(5/3))))**(2/5))*small_gamma2(x1)
    print(f'tilde part2: {part2}')
    return part1 + part2

# TODO: Figure out why p2_bar is causing the control input u to blow up
#       The system is actually relatively stable without the control input
def p2_bar(x1, tuning_parameter):
    # again, trying to split up the composition of the function for "cleanliness"
    part1 = (5/8)*((9/4)**(3/5))*((10/7)**(8/5))
    print(f'bar part1: {part1}')
    x2_star_derivative = np.abs((-smoothing_function(tuning_parameter)**(5/3))*(switching_rule(tuning_parameter)**(5/3))*((3.07002*(x1))/(np.abs(x1)**(4/3))))
    print(f'x2_star_derivative: {x2_star_derivative}')
    part2 = (((h1_bar*smoothing_function(tuning_parameter)*(beta1(x1)))**(8/5)) + (small_gamma1*(x1**(2/5)))**(8/5))
    print(f'dx2_start * part2: {(x2_star_derivative**(8/5))*part2}')
    return part1*(x2_star_derivative**(8/5))*part2+(14/5)*x2_star_derivative*2.5

def beta2(x1, x2, tuning_parameter):
    print(f'p2_hat: {p2_hat}')
    print(f'p2_tilde: {p2_tilde(x1, x2, tuning_parameter)}')
    print(f'p2_bar: {p2_bar(x1, tuning_parameter)}')
    return 1 + p2_hat + p2_tilde(x1, x2, tuning_parameter) + p2_bar(x1, tuning_parameter)

def control_law(x1, x2, tuning_parameter):
    # the control law is designed as
        # u = -Gamma_2,k * (ksi2)^(1/5) * beta2(x)
    gamma2 = switching_rule(tuning_parameter)[1]
    print(f'gamma2: {gamma2}')
    ksi2 = (x2**(5/3)) + ((smoothing_function(tuning_parameter)*switching_rule(tuning_parameter)[0]*(x1**(3/5))*2.5*(np.abs(x1)**(2/5)))**(5/3))
    print(f'ksi2: {ksi2}')
    beta2x = beta2(x1, x2, tuning_parameter)
    print(f'beta2: {beta2x}')

    return -gamma2*(ksi2**(1/5))*beta2x

# from plugging in given constants, the system is found to as
# [x1_dot, x2_dot]' = [-2x1 + 1.5x2, (11/6)(x1^2) - 2(x2) + u]
def system(t, x):
    x1, x2 = x
    for k in range(0, 4):
        u = control_law(x1,x2,k)
    A = [-2*x1+1.5*x2,
         (11/6)*(x1**2)-2*x2 + u
        ]
    return A

print(control_law(1,-1.5,0))
print(control_law(1,-1.5,1))
print(control_law(1,-1.5,2))

#x0 = np.array([1.0, -1.5],dtype=np.complex128)
#time = (0, 10)
#sol = solve_ivp(system, time, x0)
#fig, ax = plt.subplots(2,1)
#ax[0].plot(sol.t, sol.y[0])
#ax[1].plot(sol.t, sol.y[1])
#plt.show()
