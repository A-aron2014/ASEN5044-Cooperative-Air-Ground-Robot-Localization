import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy import linalg
from scipy.stats import chi2
import csv

np.random.seed(100)
#Step one find the CT Jacobains to obtain the CT Linerized model
N = 1000 # MonteCarlo Sim count
n = 6 # number of state variables
alpha = 0.05 # yields 95% confidence -> Significance level
dt  = 0.1#s
L = 0.5#m
phi_g_min = -5*np.pi/12
phi_g_max = 5*np.pi/12
vg_max = 3#m/s
ugv_nom = [10,0,np.pi/2]
ugv_nom_vg = 2
ugv_nom_phi = -np.pi/18
uav_nom = [-60,0, -np.pi/2]
uav_nom_v = 12
uav_nom_w = np.pi/25
#nominal Starting States
xi_g = ugv_nom[0]
eta_g = ugv_nom[1]
vg = ugv_nom_vg
theta_g = ugv_nom[2]
xi_a = uav_nom[0]
eta_a = uav_nom[1]
va = uav_nom_v
theta_a = uav_nom[2]
phi_g = ugv_nom_phi

def update_A(x,u):
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = x
    vg, phi_g, va, w_a = u

    A = np.array([
        [0, 0, -vg*np.sin(theta_g), 0, 0,       0],
        [0, 0, vg*np.cos(theta_g) , 0, 0,       0],
        [0, 0,          0         , 0, 0,       0],
        [0, 0,          0         , 0, 0, -va*np.sin(theta_a)],
        [0, 0,          0         , 0, 0,  va*np.cos(theta_a)],
        [0, 0,          0         , 0, 0,       0]
    ])
    return A
def update_B(x,u,L):
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = x
    vg, phi_g, va, w_a = u    
    B = np.array([
        [np.cos(theta_g), 0, 0, 0],
        [np.sin(theta_g), 0, 0, 0],
        [np.tan(phi_g)/L, (vg/L) * (1/np.cos(phi_g))**2, 0, 0],
        [0,0, np.cos(theta_a) ,0 ],
        [0, 0, np.sin(theta_a) , 0],
        [0, 0, 0, 1]
    ])
    return B
def update_H(x,u):
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = x
    vg, phi_g, va, w_a = u
    diff_eta_ag = eta_a-eta_g
    diff_xi_ag = xi_a-xi_g
    diff_eta_ga = eta_g-eta_a
    diff_xi_ga = xi_g-xi_a
    X = (diff_eta_ag)/(diff_xi_ag)
    Y = (diff_eta_ga)/(diff_xi_ga)
    ugv_uav_dist = np.sqrt(diff_xi_ag**2 + diff_eta_ag**2)
    H = np.array([
        [diff_eta_ag/((diff_xi_ag**2)*(1+X)), -1/(diff_xi_ag*(1+X)), -1, -diff_eta_ag/((diff_xi_ag**2)*(1+X)), 1/ ((diff_xi_ag)*(1+X)), 0],
        [-2*diff_xi_ag/ugv_uav_dist, -2*diff_eta_ag/ugv_uav_dist, 0 , 2*diff_xi_ag/ugv_uav_dist, 2*diff_eta_ag/ugv_uav_dist, 0],
        [-diff_eta_ga/((diff_xi_ga**2)*(1+Y)), 1/(diff_xi_ga*(1+Y)), 0, diff_eta_ga/((diff_xi_ga**2)*(1+Y)), -1/(diff_xi_ga*(1+Y)), -1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0]
        ])
    return H

u_nom = np.array([
    vg, 
    phi_g,
    va,
    uav_nom_w
    ])
x0 = np.array([ xi_g, eta_g,  theta_g, xi_a, eta_a, theta_a])
A = update_A(x0,u_nom)
B = update_B(x0,u_nom,L)
Omega = np.eye(n)

H = update_H(x0,u_nom)

#Step 2 linearize around an equillibrium point/Nominal Operating Point
#And find DT Linearized model Matrices
#Load CSV Data for Part 2 
with open('cooploc_data_csv\Qtrue.csv',newline='') as f:
    Qtrue = np.array(list(csv.reader(f,delimiter=',')), dtype=float)
with open('cooploc_data_csv\Rtrue.csv',newline='') as f:
    Rtrue = np.array(list(csv.reader(f,delimiter=',')), dtype=float)
with open(r"cooploc_data_csv\tvec.csv",newline='') as f:
    tvector = np.array(list(csv.reader(f,delimiter=',')), dtype=float).flatten()
with open('cooploc_data_csv\ydata.csv',newline='') as f:
    ydata = np.array(list(csv.reader(f,delimiter=',')), dtype=float)

F = expm(A*dt)
print('The F matrix is')
print(F)
a_row, a_col = A.shape[0],A.shape[1]
b_row, b_col = B.shape[0],B.shape[1]
A_hat = np.zeros((a_row+b_col, a_col+b_col))
A_hat[0:a_row,0:a_col] = A
A_hat[0:b_row,a_col:a_col+b_col] = B
M = expm(A_hat*dt)
F_check = M[0:a_row,0:a_col]
G = M[0:b_row,a_col:a_col+b_col]
print('checking the F matrix')
print(F_check)
print('the G matrix is')
print(G)
'''
As discussed in lecture the system is highly nonlinear and Time Varying
'''
'''
Need to calculate Q using Van Loans Method
'''
#Step 3 simulate linearized DT Dynamics and measurement models near the localization point, assuming reasonable initial state perturbations and no noise, measurement noise, or control input perturbations
time_steps = 1000#Min 400
t_steps = time_steps*dt
T = 100
num_measurements = 5
t = np.arange(0,t_steps,dt)
#Function to define the nonlinear Equations of Motion for Solve_IVP
def define_eom(t,x,u,L):
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = x
    vg, phi_g, va, w_a = u
    dxdt = np.array([
        vg*np.cos(theta_g),
        vg*np.sin(theta_g),
        (vg/L)*np.tan(phi_g),
        va*np.cos(theta_a),
        va*np.sin(theta_a),
        w_a
        ])
    return dxdt
#builds the nonlinsear h matrix 
def define_h(x):
    xi_g, eta_g, theta_g, xi_a, eta_a, theta_a = x

    y = np.array([
        np.atan2((eta_a-eta_g),(xi_a-xi_g))-theta_g,
        np.sqrt((xi_g - xi_a)**2 + (eta_g - eta_a)**2),
        np.atan2((eta_g-eta_a),(xi_g - xi_a)) - theta_a,
        xi_a,
        eta_a
    ])
    return y
x_nom = x0
#initial Perturbation Conditions
x_init_perturb = np.array([0,1,0,0,0,0.1])
x_nominal = np.zeros((time_steps+1,n))
x0_nonlinear = x0 + x_init_perturb
tspan = (0,T)
x_estimated = np.zeros((time_steps,n))
y_estimated = np.zeros((time_steps,num_measurements))
y_nonlinear = np.zeros((time_steps,num_measurements))
#use this to determine the perturbation dynamics for every timestep
x_nominal_sln = solve_ivp(lambda t, x: define_eom(t,x,u_nom,L),tspan,x0, method='RK45', t_eval=t)
x_nominal = x_nominal_sln.y.T

#Integrating the nonlinear dynamic of the model
nonlinear_x_sln = solve_ivp(lambda t, x: define_eom(t,x,u_nom,L),tspan,x0_nonlinear, method='RK45', t_eval=t)
x_nonlinear = nonlinear_x_sln.y.T

#integrating the nonlinear dynamics of the measurement model
# nonlinear_y_sln = solve_ivp(lambda t, x: define_h(t,x),tspan,x0_nonlinear  ,method='RK45',t_eval=t)
# y_nonlinear = nonlinear_y_sln.y.T

Omega_dt = dt*Omega
#initialize original y
yk = np.zeros((time_steps+1,num_measurements))
x_perturbation_vector = np.zeros((time_steps+1,n))
x_perturbation_vector[0] = x_init_perturb
delta_x = np.zeros((time_steps+1,n))
yk = H@x0_nonlinear

for i in range(time_steps):
    delta_x[i] = x_nonlinear[i] - x_nominal[i]
    A = update_A(x_nominal[i,:],u_nom)
    B = update_B(x_nominal[i,:],u_nom,L)
    H = update_H(x_nominal[i,:], u_nom)
    F_dt = np.eye(A.shape[0]) + dt*A
    G_dt = dt*B
    x_perturbation_vector[i+1] = F_dt@ x_perturbation_vector[i] #+ G_dt@u_nom
    #xk = np.linalg.matrix_power(F_dt,i) @ x_perturb + G_dt @ u_nom
    # x_estimated[i] = F_dt@x_perturbation_vector[i,:] + G_dt@ u_nom
    # y_estimated[i] = H @ x_perturbation_vector[i,:] + define_h(x_nominal[i,:]) #Adding h(x_nominal to H~ calculation of the perturbation to get the next measurement)
    x_estimated[i] = x_nominal[i] + F_dt@delta_x[i]+ G_dt@ u_nom
    y_estimated[i] = define_h(x_nominal[i])
    y_nonlinear[i] = define_h(x_nonlinear[i])
    #y_estimated[i] = H @delta_x[i] + y_nonlinear[i] #define_h(x_nominal[i,:]) #Adding h(x_nominal to H~ calculation of the perturbation to get the next measurement)

def wrap_angles(theta):
    return (theta + np.pi) %(2*np.pi) - np.pi

theta_g_bounded = wrap_angles(x_estimated[:,2])
theta_a_bounded = wrap_angles(x_estimated[:,5])

theta_g_bounded_nl = wrap_angles(x_nonlinear[:,2])
theta_a_bounded_nl = wrap_angles(x_nonlinear[:,5])

azimuth_bounded = wrap_angles(y_estimated[:,0])
nonlinear_azimuth_bounded = wrap_angles(y_nonlinear[:,0])

fig1,axs1 = plt.subplots(6,1, figsize=(10,10))

#x_estimated = x_estimated.T.flatten()
axs1[0].plot(t,x_estimated[:,0], label="linearized")
axs1[0].plot(t,x_nonlinear[:,0],'r--', label="True")
axs1[0].set_ylabel(r'$\xi_g$ [m]')
axs1[0].set_xlabel('Time [s]')
axs1[0].legend()
axs1[1].plot(t,x_estimated[:,1], label="linearized")
axs1[1].plot(t,x_nonlinear[:,1],'r--', label="True")
axs1[1].set_ylabel(r'$\eta_g$ [m]')
axs1[1].set_xlabel('Time [s]')
axs1[1].legend()
axs1[2].plot(t,theta_g_bounded, label="linearized")
axs1[2].plot(t,theta_g_bounded_nl,'r--', label="True")
axs1[2].set_ylabel(r'$\theta_g$ [m]')
axs1[2].set_xlabel('Time [s]')
axs1[2].legend()
axs1[3].plot(t,x_estimated[:,3], label="linearized")
axs1[3].plot(t,x_nonlinear[:,3],'r--', label="True")
axs1[3].set_ylabel(r'$\xi_a$ [m]')
axs1[3].set_xlabel('Time [s]')
axs1[3].legend()
axs1[4].plot(t,x_estimated[:,4], label="linearized")
axs1[4].plot(t,x_nonlinear[:,4],'r--', label="True")
axs1[4].set_ylabel(r'$\eta_a$ [m]')
axs1[4].set_xlabel('Time [s]')
axs1[4].legend()
axs1[5].plot(t,theta_a_bounded, label="linearized")
axs1[5].plot(t,theta_a_bounded_nl,'r--', label="True")
axs1[5].set_ylabel(r'$\theta_a$ [rad]')
axs1[5].set_xlabel('Time [s]')
axs1[5].legend()
fig1.suptitle("Linearized States Vs Nonlinear Dynamics")
fig2,axs2 = plt.subplots(5,1, figsize=(10,10))
# axs2[0,0].plot(tvector,y_estimated[:,0], label="linearized")
# axs2[0,0].plot(tvector,ydata[:,0],label = "True")
axs2[0].plot(tvector[:len(y_estimated)], azimuth_bounded, label="linearized")
axs2[0].plot(tvector, ydata[0,:],'r--', label="True")
axs2[0].plot(tvector[:len(y_nonlinear)], nonlinear_azimuth_bounded,'g--', label="Nonlinear Model")
axs2[0].set_ylabel(r'$\text{bearing } a \to g$')
axs2[0].set_xlabel('Time [s]')
axs2[0].legend()
axs2[1].plot(tvector[:len(y_estimated)],y_estimated[:,1],label = "linearized")
axs2[1].plot(tvector,ydata[1,:],'r--', label = "True")
axs2[1].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,1],'g--', label="Nonlinear Model")
axs2[1].set_ylabel(r'range')
axs2[1].set_xlabel('Time [s]')
axs2[1].legend()
axs2[2].plot(tvector[:len(y_estimated)],y_estimated[:,2], label = "linearized")
axs2[2].plot(tvector,ydata[2,:],'r--', label = "True")
axs2[2].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,2],'g--', label="Nonlinear Model")
axs2[2].set_ylabel(r'$\text{bearing } g \to a$')
axs2[2].set_xlabel('Time [s]')
axs2[2].legend()
axs2[3].plot(tvector[:len(y_estimated)],y_estimated[:,3], label = "linearized")
axs2[3].plot(tvector,ydata[3,:],'r--', label = "True")
axs2[3].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,3],'g--', label="Nonlinear Model")
axs2[3].set_ylabel(r'UAV $\xi$ (GPS)')
axs2[3].set_xlabel('Time [s]')
axs2[3].legend()
axs2[4].plot(tvector[:len(y_estimated)],y_estimated[:,4], label = "linearized")
axs2[4].plot(tvector,ydata[4,:],'r--', label = "True")
axs2[4].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,4],'g--', label="Nonlinear Model")
axs2[4].set_ylabel(r'UAV $\eta$ (GPS)')
axs2[4].set_xlabel('Time [s]')
axs2[4].legend()
fig2.suptitle("Approximate Linearized Model Data Vs True Measurement Data")
fig3,axs3 = plt.subplots(6,1, figsize=(10,10))
axs3[0].plot(tvector[:len(x_perturbation_vector)], x_perturbation_vector[:,0], label="linearized")

#axs3[0].plot(tvector[:len(delta_x)], delta_x[:,0],'r--', label="True")
axs3[0].set_ylabel(r'$\delta \xi_g$')
axs3[0].set_xlabel('Time [s]')

axs3[1].plot(tvector[:len(x_perturbation_vector)], x_perturbation_vector[:,1], label="linearized")
#axs3[1].plot(tvector[:len(delta_x)], delta_x[:,1],'r--', label="True")
axs3[1].set_ylabel(r'$\delta \eta_g$')
axs3[1].set_xlabel('Time [s]')

axs3[2].plot(tvector[:len(x_perturbation_vector)], x_perturbation_vector[:,2], label="linearized")
#axs3[2].plot(tvector[:len(delta_x)], delta_x[:,2],'r--', label="True")
axs3[2].set_ylabel(r'$\delta \theta_g$')
axs3[2].set_xlabel('Time [s]')

axs3[3].plot(tvector[:len(x_perturbation_vector)], x_perturbation_vector[:,3], label="linearized")
#axs3[3].plot(tvector[:len(delta_x)], delta_x[:,3],'r--', label="True")
axs3[3].set_ylabel(r'$\delta \xi_a$')
axs3[3].set_xlabel('Time [s]')

axs3[4].plot(tvector[:len(x_perturbation_vector)], x_perturbation_vector[:,4], label="linearized")
#axs3[4].plot(tvector[:len(delta_x)], delta_x[:,4],'r--', label="True")
axs3[4].set_ylabel(r'$\delta \eta_a$')
axs3[4].set_xlabel('Time [s]')

axs3[5].plot(tvector[:len(x_perturbation_vector)], x_perturbation_vector[:,5], label="linearized")
#axs3[5].plot(tvector[:len(delta_x)], delta_x[:,5],'r--', label="True")
axs3[5].set_ylabel(r'$\delta \theta_a$')
axs3[5].set_xlabel('Time [s]')
fig3.suptitle("Linearized Perturbations")

plt.show()

#Perturbation Dynamics
#F~ = I + dt*A~evaluated at nominal trajectory
#G~ = dt*B~ evaluated at nominal trajectory ->Does nominal trajectory change?
#Omega~ = dt*Gamma
#H~ = h evaluated at the nominal trajectory

def calc_F(A, dt):
    F = np.eye(A.shape[0]) + A*dt
    return F

def calc_G(B,dt):
    G = B*dt
    return G
def calc_Omega(Gamma,dt):
    Omega = Gamma*dt
    return Omega

#x_perturb_dot = A~x_perturb + B~u_perturb + Gamma*w~
#y_perturb_dot = C~*x_perturb + v

'''
Here I will put the LKF Code

'''
#NEES Epsilon bar statistics should approach expected value of chi square distribution which is n number of state variables as # of monte carlo sims goes to infinity

#For NIS Should approach p number of observations
#Mismodeling error in Q_kf if there is a huge error in process noise should be able to detect with 10-15 simulations but if you have a small error you need more simulations


'''
PART 5: THE EKF
'''

'''
old settings json
    "security.workspace.trust.untrustedFiles": "open",
    "python.createEnvironment.trigger": "off",
    "git.openRepositoryInParentFolders": "never",
    "editor.hover.enabled": false,
    "editor.parameterHints.enabled": true,
    "editor.tabCompletion": "on",
    "editor.quickSuggestions": {
        "comments": "on",
        "strings": "on"
    }


'''
for k_ekf in range(time_steps):
    x_nonlinear