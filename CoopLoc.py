import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy import linalg
from scipy.stats.distributions import chi2
from scipy.linalg import cho_factor, cho_solve
import csv

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

def calc_F(A, dt):
    F = np.eye(A.shape[0]) + A*dt
    return F

def calc_G(B,dt):
    G = B*dt
    return G
def calc_Omega(Gamma,dt):
    Omega = Gamma*dt
    return Omega
def wrap_angles(theta):
    return (theta + np.pi) %(2*np.pi) - np.pi

def run_dynamics_model(time_steps,t):
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

    #initialize original y
    yk = np.zeros((time_steps+1,num_measurements))
    x_perturbation_vector = np.zeros((time_steps+1,n))
    x_perturbation_vector[0] = x_init_perturb
    delta_x = np.zeros((time_steps+1,n))

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

    theta_g_bounded = wrap_angles(x_estimated[:,2])
    theta_a_bounded = wrap_angles(x_estimated[:,5])

    theta_g_bounded_nl = wrap_angles(x_nonlinear[:,2])
    theta_a_bounded_nl = wrap_angles(x_nonlinear[:,5])

    azimuth_bounded = wrap_angles(y_estimated[:,0])
    nonlinear_azimuth_bounded = wrap_angles(y_nonlinear[:,0])

    fig1,axs1 = plt.subplots(6,1, figsize=(10,10))

    #x_estimated = x_estimated.T.flatten()
    # axs1[0].plot(t,x_estimated[:,0], label="linearized")
    # axs1[0].plot(t,x_nonlinear[:,0],'r--', label="True")
    # axs1[0].set_ylabel(r'$\xi_g$ [m]')
    # axs1[0].set_xlabel('Time [s]')
    # axs1[0].legend()
    # axs1[1].plot(t,x_estimated[:,1], label="linearized")
    # axs1[1].plot(t,x_nonlinear[:,1],'r--', label="True")
    # axs1[1].set_ylabel(r'$\eta_g$ [m]')
    # axs1[1].set_xlabel('Time [s]')
    # axs1[1].legend()
    # axs1[2].plot(t,theta_g_bounded, label="linearized")
    # axs1[2].plot(t,theta_g_bounded_nl,'r--', label="True")
    # axs1[2].set_ylabel(r'$\theta_g$ [m]')
    # axs1[2].set_xlabel('Time [s]')
    # axs1[2].legend()
    # axs1[3].plot(t,x_estimated[:,3], label="linearized")
    # axs1[3].plot(t,x_nonlinear[:,3],'r--', label="True")
    # axs1[3].set_ylabel(r'$\xi_a$ [m]')
    # axs1[3].set_xlabel('Time [s]')
    # axs1[3].legend()
    # axs1[4].plot(t,x_estimated[:,4], label="linearized")
    # axs1[4].plot(t,x_nonlinear[:,4],'r--', label="True")
    # axs1[4].set_ylabel(r'$\eta_a$ [m]')
    # axs1[4].set_xlabel('Time [s]')
    # axs1[4].legend()
    # axs1[5].plot(t,theta_a_bounded, label="linearized")
    # axs1[5].plot(t,theta_a_bounded_nl,'r--', label="True")
    # axs1[5].set_ylabel(r'$\theta_a$ [rad]')
    # axs1[5].set_xlabel('Time [s]')
    # axs1[5].legend()
    # fig1.suptitle("Linearized States Vs Nonlinear Dynamics")

    # fig2,axs2 = plt.subplots(5,1, figsize=(10,10))
    # axs2[0].plot(tvector[:len(y_estimated)], azimuth_bounded, label="linearized")
    # axs2[0].plot(tvector, ydata[0,:],'r--', label="True")
    # axs2[0].plot(tvector[:len(y_nonlinear)], nonlinear_azimuth_bounded,'g--', label="Nonlinear Model")
    # axs2[0].set_ylabel(r'$\text{bearing } a \to g$')
    # axs2[0].set_xlabel('Time [s]')
    # axs2[0].legend()
    # axs2[1].plot(tvector[:len(y_estimated)],y_estimated[:,1],label = "linearized")
    # axs2[1].plot(tvector,ydata[1,:],'r--', label = "True")
    # axs2[1].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,1],'g--', label="Nonlinear Model")
    # axs2[1].set_ylabel(r'range')
    # axs2[1].set_xlabel('Time [s]')
    # axs2[1].legend()
    # axs2[2].plot(tvector[:len(y_estimated)],y_estimated[:,2], label = "linearized")
    # axs2[2].plot(tvector,ydata[2,:],'r--', label = "True")
    # axs2[2].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,2],'g--', label="Nonlinear Model")
    # axs2[2].set_ylabel(r'$\text{bearing } g \to a$')
    # axs2[2].set_xlabel('Time [s]')
    # axs2[2].legend()
    # axs2[3].plot(tvector[:len(y_estimated)],y_estimated[:,3], label = "linearized")
    # axs2[3].plot(tvector,ydata[3,:],'r--', label = "True")
    # axs2[3].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,3],'g--', label="Nonlinear Model")
    # axs2[3].set_ylabel(r'UAV $\xi$ (GPS)')
    # axs2[3].set_xlabel('Time [s]')
    # axs2[3].legend()
    # axs2[4].plot(tvector[:len(y_estimated)],y_estimated[:,4], label = "linearized")
    # axs2[4].plot(tvector,ydata[4,:],'r--', label = "True")
    # axs2[4].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,4],'g--', label="Nonlinear Model")
    # axs2[4].set_ylabel(r'UAV $\eta$ (GPS)')
    # axs2[4].set_xlabel('Time [s]')
    # axs2[4].legend()
    # fig2.suptitle("Approximate Linearized Model Data Vs True Measurement Data")

    # fig3,axs3 = plt.subplots(6,1, figsize=(10,10))
    # axs3[0].plot(tvector[:len(x_perturbation_vector)], x_perturbation_vector[:,0], label="linearized")
    # axs3[0].set_ylabel(r'$\delta \xi_g$')
    # axs3[0].set_xlabel('Time [s]')

    # axs3[1].plot(tvector[:len(x_perturbation_vector)], x_perturbation_vector[:,1], label="linearized")
    # axs3[1].set_ylabel(r'$\delta \eta_g$')
    # axs3[1].set_xlabel('Time [s]')

    # axs3[2].plot(tvector[:len(x_perturbation_vector)], x_perturbation_vector[:,2], label="linearized")
    # axs3[2].set_ylabel(r'$\delta \theta_g$')
    # axs3[2].set_xlabel('Time [s]')

    # axs3[3].plot(tvector[:len(x_perturbation_vector)], x_perturbation_vector[:,3], label="linearized")
    # axs3[3].set_ylabel(r'$\delta \xi_a$')
    # axs3[3].set_xlabel('Time [s]')

    # axs3[4].plot(tvector[:len(x_perturbation_vector)], x_perturbation_vector[:,4], label="linearized")
    # axs3[4].set_ylabel(r'$\delta \eta_a$')
    # axs3[4].set_xlabel('Time [s]')

    # axs3[5].plot(tvector[:len(x_perturbation_vector)], x_perturbation_vector[:,5], label="linearized")
    # axs3[5].set_ylabel(r'$\delta \theta_a$')
    # axs3[5].set_xlabel('Time [s]')
    # fig3.suptitle("Linearized Perturbations")

    # plt.show()
    return x_nonlinear, x_estimated, y_nonlinear, y_estimated
#Perturbation Dynamics
#F~ = I + dt*A~evaluated at nominal trajectory
#G~ = dt*B~ evaluated at nominal trajectory ->Does nominal trajectory change?
#Omega~ = dt*Gamma
#H~ = h evaluated at the nominal trajectory



#x_perturb_dot = A~x_perturb + B~u_perturb + Gamma*w~
#y_perturb_dot = C~*x_perturb + v

'''
Here I will put the LKF Code

'''
def run_lkf():
    pass

#NEES Epsilon bar statistics should approach expected value of chi square distribution which is n number of state variables as # of monte carlo sims goes to infinity

#For NIS Should approach p number of observations
#Mismodeling error in Q_kf if there is a huge error in process noise should be able to detect with 10-15 simulations but if you have a small error you need more simulations


'''
PART 5: THE EKF
'''
#Truth Run
def run_ekf(ydata,tvector,Q,Gamma,R,x0):

    x_err = np.zeros((time_steps+1,n))
    sigma = np.zeros((time_steps+1,n))
    x_estimated = np.zeros((time_steps+1,n))#linearized a priori x state
    x_total_est = np.zeros((time_steps+1,n))#a posteriori state update
    P_posteriori = np.eye(n) * 1e-6 #Updated Covariance
    P_apriori = P_posteriori.copy() #Prior Covariance
    P_update = [P_posteriori]
    x_apriori = np.zeros((time_steps+1,n))#prior estimate of x
    y_apriori = np.zeros((time_steps+1,num_measurements))
    innovations = np.zeros((time_steps+1,num_measurements))

    epsilon = np.zeros(time_steps+1)

    x_total_est[0] = x0   
    Omega = calc_Omega(Gamma,dt)
    tspan = (0,T)
    #Integrating the nonlinear dynamic of the model
    x_true_NL = solve_ivp(lambda t, x: define_eom(t,x,u_nom,L),tspan,x_total_est[0], method='RK45', t_eval=t)
    x_true = x_true_NL.y.T
    
    for k in range(0,time_steps):
        tspan_ekf = (tvector[k], tvector[k+1])
        if not np.all(np.isfinite(x_total_est[k])):
            raise ValueError(f"Non-finite state at step {k}: {x_total_est[k]}")

        # Compute x_nonlinear for timesteps k to k+1 using EOM
        solution = solve_ivp(lambda t, x: define_eom(t,x,u_nom,L),tspan_ekf,x_total_est[k], method='RK45', t_eval=[tvector[k+1]])
        x_apriori[k+1] = solution.y[:,-1]

        if not np.all(np.isfinite(x_apriori[k+1])):
            print(f"Warning: x_apriori at step {k+1} has NaN/inf: {x_apriori[k+1]}")

        #take compute the estimate version of this by taking the best guess and then plugging that into the euler form CT -> DT dynamios then sum them
        #Use estimated x to compute P
        A = update_A(x_total_est[k],u_nom)
        B = update_B(x_apriori[k],u_nom,L)
        x_estimated[k+1] = x_apriori[k] + dt * (A@x_apriori[k]+ B@ u_nom)
        F = calc_F(A,dt)
        G = calc_G(B,dt)

        P_apriori = F @ P_update[-1] @ F.T + Omega @ Q @ Omega.T

        #calculate y_hat_a priori by evaluating h using x_nonlinear
        y_apriori[k+1] = define_h(x_apriori[k+1])
        #Build H using linear
        H = update_H(x_apriori[k+1],u_nom)
        #calculate innovation
        innovations[k+1] = ydata[:,k+1] - y_apriori[k+1]
        #calculate K Kalman Gain Matrix
        S = H @ P_apriori @ H.T + R
        K_kf = P_apriori @ H.T @ np.linalg.inv(S)
        #Calculate Updated total estimate
        x_total_est[k+1] = x_apriori[k+1] + K_kf @ innovations[k+1]
        #Calculate Updated Covariance
        P_posteriori = (np.eye(P_apriori.shape[0]) - K_kf @ H) @ P_apriori
        P_update.append(P_posteriori)
        sigma[k] = np.sqrt(np.diag(P_posteriori))
        x_err[k] = x_true[k] - x_total_est[k]
        epsilon[k] = x_err[k].T @ P_posteriori @ x_err[k]

    upper_bounds = 2*sigma
    lower_bounds = -2*sigma
    fig1,axs1 = plt.subplots(6,1, figsize=(10,10))
    theta_g_bounded = wrap_angles(x_total_est[:,2])
    theta_a_bounded = wrap_angles(x_total_est[:,5])

    theta_g_bounded_nl = wrap_angles(x_true[:,2])
    theta_a_bounded_nl = wrap_angles(x_true[:,5])

    azimuth_bounded = wrap_angles(y_apriori[:,0])
    nonlinear_azimuth_bounded = wrap_angles(innovations[:,0])
    x_estimated = x_estimated.T.flatten()
    axs1[0].plot(t,x_total_est[:time_steps,0], label="linearized")
    axs1[0].plot(t,x_true[:time_steps,0],'r--', label="True")
    axs1[0].set_ylabel(r'$\xi_g$ [m]')
    axs1[0].set_xlabel('Time [s]')
    axs1[0].legend()
    axs1[1].plot(t,x_total_est[:time_steps,1], label="linearized")
    axs1[1].plot(t,x_true[:time_steps,1],'r--', label="True")
    axs1[1].set_ylabel(r'$\eta_g$ [m]')
    axs1[1].set_xlabel('Time [s]')
    axs1[1].legend()
    axs1[2].plot(t,theta_g_bounded[:time_steps], label="linearized")
    axs1[2].plot(t,theta_g_bounded_nl[:time_steps],'r--', label="True")
    axs1[2].set_ylabel(r'$\theta_g$ [m]')
    axs1[2].set_xlabel('Time [s]')
    axs1[2].legend()
    axs1[3].plot(t,x_total_est[:time_steps,3], label="linearized")
    axs1[3].plot(t,x_true[:time_steps,3],'r--', label="True")
    axs1[3].set_ylabel(r'$\xi_a$ [m]')
    axs1[3].set_xlabel('Time [s]')
    axs1[3].legend()
    axs1[4].plot(t,x_total_est[:time_steps,4], label="linearized")
    axs1[4].plot(t,x_true[:time_steps,4],'r--', label="True")
    axs1[4].set_ylabel(r'$\eta_a$ [m]')
    axs1[4].set_xlabel('Time [s]')
    axs1[4].legend()
    axs1[5].plot(t,theta_a_bounded[:time_steps], label="linearized")
    axs1[5].plot(t,theta_a_bounded_nl[:time_steps],'r--', label="True")
    axs1[5].set_ylabel(r'$\theta_a$ [rad]')
    axs1[5].set_xlabel('Time [s]')
    axs1[5].legend()
    fig1.suptitle("Linearized States Vs Nonlinear Dynamics")
    
    fig2,axs2 = plt.subplots(5,1, figsize=(10,10))
    axs2[0].plot(tvector[:len(y_apriori)], azimuth_bounded, label="Estimated")
    axs2[0].plot(tvector, ydata[0,:],'r--', label="True")
    #axs2[0].plot(tvector[:len(y_nonlinear)], nonlinear_azimuth_bounded,'g--', label="Nonlinear Model")
    axs2[0].set_ylabel(r'$\text{bearing } a \to g$')
    axs2[0].set_xlabel('Time [s]')
    axs2[0].legend()
    axs2[1].plot(tvector[:len(y_apriori)],y_apriori[:,1],label = "Estimated")
    axs2[1].plot(tvector,ydata[1,:],'r--', label = "True")
    #axs2[1].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,1],'g--', label="Nonlinear Model")
    axs2[1].set_ylabel(r'range')
    axs2[1].set_xlabel('Time [s]')
    axs2[1].legend()
    axs2[2].plot(tvector[:len(y_apriori)],y_apriori[:,2], label = "Estimated")
    axs2[2].plot(tvector,ydata[2,:],'r--', label = "True")
    #axs2[2].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,2],'g--', label="Nonlinear Model")
    axs2[2].set_ylabel(r'$\text{bearing } g \to a$')
    axs2[2].set_xlabel('Time [s]')
    axs2[2].legend()
    axs2[3].plot(tvector[:len(y_apriori)],y_apriori[:,3], label = "Estimated")
    axs2[3].plot(tvector,ydata[3,:],'r--', label = "True")
    #axs2[3].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,3],'g--', label="Nonlinear Model")
    axs2[3].set_ylabel(r'UAV $\xi$ (GPS)')
    axs2[3].set_xlabel('Time [s]')
    axs2[3].legend()
    axs2[4].plot(tvector[:len(y_apriori)],y_apriori[:,4], label = "Estimated")
    axs2[4].plot(tvector,ydata[4,:],'r--', label = "True")
    #axs2[4].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,4],'g--', label="Nonlinear Model")
    axs2[4].set_ylabel(r'UAV $\eta$ (GPS)')
    axs2[4].set_xlabel('Time [s]')
    axs2[4].legend()
    fig2.suptitle("Approximate Linearized Model Data Vs True Measurement Data")


    fig3,axs3 = plt.subplots(6,1, figsize=(10,10))
    axs3[0].plot(t,x_err[:time_steps,0], label="linearized")
    axs3[0].plot(t,upper_bounds[:time_steps,0],'r--', label="True")
    axs3[0].plot(t,lower_bounds[:time_steps,0],'r--', label="True")
    axs3[0].set_ylabel(r'$\xi_g$ [m]')
    axs3[0].set_xlabel('Time [s]')
    axs3[0].legend()
    axs3[1].plot(t,x_err[:time_steps,1], label="linearized")
    axs3[1].plot(t,upper_bounds[:time_steps,1],'r--', label="True")
    axs3[1].plot(t,lower_bounds[:time_steps,1],'r--', label="True")
    axs3[1].set_ylabel(r'$\eta_g$ [m]')
    axs3[1].set_xlabel('Time [s]')
    axs3[1].legend()
    axs3[2].plot(t,x_err[:time_steps,2], label="linearized")
    axs3[2].plot(t,upper_bounds[:time_steps,2],'r--', label="True")
    axs3[2].plot(t,lower_bounds[:time_steps,2],'r--', label="True")
    axs3[2].set_ylabel(r'$\theta_g$ [m]')
    axs3[2].set_xlabel('Time [s]')
    axs3[2].legend()
    axs3[3].plot(t,x_err[:time_steps,3], label="linearized")
    axs3[3].plot(t,upper_bounds[:time_steps,3],'r--', label="True")
    axs3[3].plot(t,lower_bounds[:time_steps,3],'r--', label="True")
    axs3[3].set_ylabel(r'$\xi_a$ [m]')
    axs3[3].set_xlabel('Time [s]')
    axs3[3].legend()
    axs3[4].plot(t,x_err[:time_steps,4], label="linearized")
    axs3[4].plot(t,upper_bounds[:time_steps,4],'r--', label="True")
    axs3[4].plot(t,lower_bounds[:time_steps,4],'r--', label="True")
    axs3[4].set_ylabel(r'$\eta_a$ [m]')
    axs3[4].set_xlabel('Time [s]')
    axs3[4].legend()
    axs3[5].plot(t,x_err[:time_steps,5], label="linearized")
    axs3[5].plot(t,upper_bounds[:time_steps,5],'r--', label="True")
    axs3[5].plot(t,lower_bounds[:time_steps,5],'r--', label="True")
    axs3[5].set_ylabel(r'$\theta_a$ [rad]')
    axs3[5].set_xlabel('Time [s]')
    axs3[5].legend()
    fig3.suptitle("State Errors and 2Sigma Bounds")

    fig4,axs4 = plt.subplots(1,1, figsize=(10,10))
    axs4.plot(t,epsilon[:time_steps], label = 'Chi Square NEES')
    axs4.set_ylabel('Chi Square Error')
    axs4.set_xlabel('Time [s]')
    axs4.legend()
    fig4.suptitle("NEES")
    plt.show()


    return x_total_est, y_apriori, P_posteriori, epsilon

def ekf_truth(tvector,Q,Gamma,R, X0, P0, time_steps, dt, u_nom, L, rng):
    x = np.zeros((time_steps+1, x0.size)); x[0] = x0
    z = np.zeros((time_steps+1, R.shape[0]))
    for k in range(time_steps):
        w = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
        v = np.random.multivariate_normal(np.zeros(R.shape[0]), R)

        sol_truth = solve_ivp(lambda t, x: define_eom(t, x, u_nom, L),
                    (tvector[k], tvector[k+1]),
                    x[k], method='RK45', t_eval=[tvector[k+1]])
        
        x[k+1] = sol_truth.y[:, -1] + w
        z[k+1] = define_h(x[k+1]) + v
    return x,z

def run_ekf_noisy(tvector,x0,P0,x_truth,y_truth, Q_kf, Gamma, R_kf, time_steps, dt, u_nom, L):
    #Total State update of X  
    x_estimated = np.zeros((time_steps+1,n))#linearized a priori x state
    P_posteriori = P0 #Updated Covariance
    P_apriori = P_posteriori.copy() #Prior Covariance
    x_apriori = np.zeros((time_steps+1,n))#prior estimate of x

    #noisy measurements
    y_hat  = np.zeros((time_steps+1,num_measurements))
    innovations = np.zeros((time_steps+1,num_measurements))
    #KF Data to store 
    P_list = np.zeros((time_steps+1,n,n))
    S_list = np.zeros((time_steps+1,num_measurements,num_measurements))
    sigma = np.zeros((time_steps+1,n))
    e = np.zeros((time_steps+1,n)) 
    Omega = calc_Omega(Gamma,dt)
    P_list[0,:,:] = P0.copy()

    # Consistency stats
    nees = np.zeros(time_steps + 1)
    nis  = np.zeros(time_steps + 1)
    x_estimated[0] = x0
    for k in range(0,time_steps):
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        #Prediction Step of the Filter
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        # Prediction propagation from previous estimate (or same sol if you intend)
        sol_pred = solve_ivp(lambda t, x: define_eom(t, x, u_nom, L),
                            (tvector[k], tvector[k+1]),
                            x_estimated[k], method='RK45', t_eval=[tvector[k+1]])
        x_apriori[k+1] = sol_pred.y[:, -1]
        #Verify that the calculated state is finite otherwise it will ruin the filter
        if not np.all(np.isfinite(x_apriori[k+1])):
            print(f"Warning: x_apriori at step {k+1} has NaN/inf: {x_apriori[k+1]}")
        #take compute the estimate version of this by taking the best guess and then plugging that into the euler form CT -> DT dynamios then sum them
        A = update_A(x_apriori[k+1],u_nom)
        #x_linearized = x_apriori[k] + dt * (A@x_apriori[k]+ B@ u_nom)
        F = calc_F(A,dt)

        #calculate P_apriori
        P_apriori = F @ P_list[k] @ F.T + Omega @ Q_kf @ Omega.T

        #----------------------------------------------------------------------------------------------------------------------------------------------------
        #Update Step of the Filter
        #----------------------------------------------------------------------------------------------------------------------------------------------------
        #calculate y_hat a priori by evaluating h using x_apriori
        y_hat[k+1] = define_h(x_apriori[k+1])
        #Build H
        H = update_H(x_apriori[k+1],u_nom)
        #calculate innovation
        innovations[k+1] = y_truth[k+1] - y_hat[k+1]
        #calculate K Kalman Gain Matrix
        S = H @ P_apriori @ H.T + R_kf
        S_list[k+1] = S
        K_kf = np.linalg.solve(S, (P_apriori @ H.T).T).T#P_apriori @ H.T @ np.linalg.inv(S)
        #Calculate Updated total estimate
        x_estimated[k+1] = x_apriori[k+1] + K_kf @ innovations[k+1]
        #Calculate Updated Covariance
        #P_posteriori = (np.eye(P_apriori.shape[0]) - K_kf @ H) @ P_apriori
        I = np.eye(P_apriori.shape[0])
        P_posteriori = (I - K_kf @ H) @ P_apriori @ (I - K_kf @ H).T + K_kf @ R_kf @ K_kf.T
        P_list[k+1,:,:] = P_posteriori
        sigma[k] = np.sqrt(np.diag(P_posteriori))
        #-------------------------------------------------------------------------------------------------------------------------------------------------------
        #Consistency Stats
        #-------------------------------------------------------------------------------------------------------------------------------------------------------
        e[k+1] = x_truth[k+1] - x_estimated[k+1]
        cS = cho_factor(S,lower=True)
        cP = cho_factor(P_posteriori,lower = True)
        nees[k+1] = e[k+1].T @ cho_solve(cP,e[k+1]) #np.linalg.inv(P_posteriori) @ e[k+1]
        nis[k+1] = innovations[k+1].T @ cho_solve(cS,innovations[k+1]) #np.linalg.inv(S) @innovations[k+1]
    return x_estimated, e, P_list, innovations, S_list, sigma, y_hat, nees, nis

def compute_nees(xs, xhat, P):
    K = xs.shape[0] - 1
    nees = np.zeros(K+1)
    for k in range(K+1):
        e = xs[k] - xhat[k]
        cP = cho_factor(P[k])
        nees[k] = e.T @ cho_solve(cP, e)
    return nees

def compute_nis(y_seq, S_seq):
    K = y_seq.shape[0]
    nis = np.zeros(K)
    for k in range(K):
        cS = cho_factor(S_seq[k])
        nis[k] = y_seq[k].T @ cho_solve(cS, y_seq[k])
    return nis

def chi_bounds(dim, N, beta=0.95):
    alpha = 1 - beta
    low = chi2.ppf(alpha/2, df=dim * N) / N
    high = chi2.ppf(1 - alpha/2, df=dim * N) / N
    return low, high

if __name__ == "__main__":
    rng = np.random.default_rng(100)
    N = 100 # MonteCarlo Sim count
    n = 6 # number of state variables
    alpha = 0.05 # yields 95% confidence -> Significance level
    dt  = 0.1#s

    TEST_STATE = "EKF"#Set to wahwhatevertever you want to run

    #Initial Variables
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

    u_nom = np.array([
    vg, 
    phi_g,
    va,
    uav_nom_w
    ])
    x0 = np.array([ xi_g, eta_g,  theta_g, xi_a, eta_a, theta_a])
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Step one find the CT Jacobains to obtain the CT Linerized model
    #Defining initial nominal State matrices
    A = update_A(x0,u_nom)
    B = update_B(x0,u_nom,L)
    Gamma = np.eye(n)
    H = update_H(x0,u_nom)
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Load CSV Data
    with open('cooploc_data_csv\Qtrue.csv',newline='') as f:
        Qtrue = np.array(list(csv.reader(f,delimiter=',')), dtype=float)
    with open('cooploc_data_csv\Rtrue.csv',newline='') as f:
        Rtrue = np.array(list(csv.reader(f,delimiter=',')), dtype=float)
    with open(r"cooploc_data_csv\tvec.csv",newline='') as f:
        tvector = np.array(list(csv.reader(f,delimiter=',')), dtype=float).flatten()
    with open('cooploc_data_csv\ydata.csv',newline='') as f:
        ydata = np.array(list(csv.reader(f,delimiter=',')), dtype=float)
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Calculating Traditional DT State Matrices
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
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    As discussed in lecture the system is highly nonlinear and Time Varying. Therefore regular Observability and Stability checks for LTI systems will not work here
    '''
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Step 3 simulate linearized DT Dynamics and measurement models near the localization point, assuming reasonable initial state perturbations and no noise, measurement noise, or control input perturbations
    time_steps = 1000#Min 400
    t_steps = time_steps*dt
    T = 100
    num_measurements = 5
    t = np.arange(0,t_steps,dt)
    #TODO Uncomment Later
    #x_nonlinear, x_estimated, y_nonlinear, y_estimated =run_dynamics_model(time_steps,t)

        #Plot g.T @ g
    match TEST_STATE:
        case "LKF":
            #Run LKF SIM
            pass
        case "EKF":
            #Run EKF SIM
            P0 =np.diag([0.1, 0.1, 0.01, 0.1, 0.1, 0.01])
            #def ekf_truth(tvector,Q,Gamma,R, X0, P0, time_steps, dt, u_nom, L, rng):
            x_truth, y_truth = ekf_truth(tvector,Qtrue,Gamma,Rtrue,x0,P0,time_steps,dt,u_nom,L,rng)

            qw= 10
            Q_ekf = qw*Qtrue

            rw = 100
            R_ekf = rw*Rtrue
            #Run the NEES Model
            #r1_nees = chi2.ppf(alpha/2, N*n)/N
            #r2_nees = chi2.ppf(1-alpha/2,N*n)/N

            r1_nees,r2_nees = chi_bounds(n,N)
            #r1_nis = chi2.ppf(alpha/2, N*num_measurements)/N
            #r2_nis = chi2.ppf(1-alpha/2,N*num_measurements)/N
            r1_nis,r2_nis = chi_bounds(num_measurements,N)
            NEES = np.zeros((N,time_steps+1))
            NIS = np.zeros((N,time_steps+1))

            for k in range(N):
                #First MC Run to generate truth value 
                #y_meas, x_estimated, e, P_list, innovations, S_list, sigma, y_hat, nees, nis # run_ekf_noisy(tvector,x0,P0,x_truth,y_truth, Q_kf, Gamma, R_kf, time_steps, dt, u_nom, L):

                x_est, epsilon, P_hist,innovations,S_list,sigma,y_hat, nees, nis = run_ekf_noisy(tvector,x0,P0,x_truth,y_truth, Q_ekf,Gamma,Rtrue,time_steps,dt,u_nom,L)

                # for j in range(1,time_steps+1):
                #     #NEES calculations
                #     ex_k = dx_true[j,:] 
                #     Pk = P_hist[j,:,:]
                #     NEES[k,j] = ex_k @ np.linalg.inv(Pk) @ ex_k.T

                #     #NIS Calculations
                #     ey_k = innovations[j,:]
                #     S_k = S_list[j,:,:]
                #     #NIS Calc
                #     NIS[k,j] = ey_k.T @np.linalg.inv(S_k)@ey_k

                NEES[k,:]=nees
                NIS[k,:] = nis
            #Take the average of the NEES runs
            print(NIS)
            NEES_mean = np.mean(NEES, axis=0)

            #Take the average of the NIS runs
            NIS_mean = np.mean(NIS, axis=0)
            #Run the NIS Test
            fig10,axs10 = plt.subplots(1,1, figsize = (10,12))
            axs10.plot(t, NEES_mean[1:],'k', label = "Zoomed out NEES")
            #axs10.plot(t[1:100], NEES_mean[1:100], 'k', label = 'Zoomed in NEES')
            axs10.hlines(r1_nees, xmin=t[1], xmax=t[-1], colors='r',linestyles='--')
            axs10.hlines(r2_nees, xmin=t[1], xmax=t[-1], colors='g',linestyles='--')
            axs10.set_title('NEES Test Statistic Points vs Time')
            axs10.set_xlabel('Time [s]')
            axs10.set_ylabel('NEES')
            axs10.grid(True)
            fig10.tight_layout()

            fig11,axs11 = plt.subplots(1,1, figsize = (10,12))
            axs11.plot(t, NIS_mean[1:],'k', label = "Zoomed out NIS")
            #axs11.plot(t[1:100], NIS_mean[1:100], 'k', label = 'Zoomed in NIS')
            axs11.hlines(r1_nis, xmin=t[1], xmax=t[-1], colors='r',linestyles='--')
            axs11.hlines(r2_nis, xmin=t[1], xmax=t[-1], colors='g',linestyles='--')
            axs11.set_title('NIS Test Statistic Points vs Time')
            axs11.set_xlabel('Time [s]')
            axs11.set_ylabel('NIS')
            axs11.grid(True)
            fig11.tight_layout()

            fig1,axs1 = plt.subplots(6,1, figsize=(10,10))
            axs1[0].plot(t,x_est[:time_steps,0], label=r'$\xi_g$ Estimated')
            axs1[0].plot(t,x_truth[:time_steps,0],'r--', label=r'$\xi_g$ True')
            axs1[0].set_ylabel(r'$\xi_g$ [m]')
            axs1[0].set_xlabel('Time [s]')
            axs1[0].legend()
            axs1[1].plot(t,x_est[:time_steps,1], label=r'$\eta_g$ Estimated')
            axs1[1].plot(t,x_truth[:time_steps,1],'r--', label=r'$\eta_g$ True')
            axs1[1].set_ylabel(r'$\eta_g$ [m]')
            axs1[1].set_xlabel('Time [s]')
            axs1[1].legend()
            axs1[2].plot(t,x_est[:time_steps,2], label=r'$\theta_g$ Estimated')
            axs1[2].plot(t,x_truth[:time_steps,2],'r--', label=r'$\theta_g$ True')
            axs1[2].set_ylabel(r'$\theta_g$ [m]')
            axs1[2].set_xlabel('Time [s]')
            axs1[2].legend()
            axs1[3].plot(t,x_est[:time_steps,3], label=r'$\xi_a$ Estimated')
            axs1[3].plot(t,x_truth[:time_steps,3],'r--', label=r'$\xi_a$ True')
            axs1[3].set_ylabel(r'$\xi_a$ [m]')
            axs1[3].set_xlabel('Time [s]')
            axs1[3].legend()
            axs1[4].plot(t,x_est[:time_steps,4], label=r'$\eta_a$ Estimated')
            axs1[4].plot(t,x_truth[:time_steps,4],'r--', label=r'$\eta_a$ True')
            axs1[4].set_ylabel(r'$\eta_a$ [m]')
            axs1[4].set_xlabel('Time [s]')
            axs1[4].legend()
            axs1[5].plot(t,x_est[:time_steps,5], label=r'$\theta_a$ Estimated')
            axs1[5].plot(t,x_truth[:time_steps,5],'r--', label= r'$\theta_a$ True')
            axs1[5].set_ylabel(r'$\theta_a$ [rad]')
            axs1[5].set_xlabel('Time [s]')
            axs1[5].legend()
            fig1.suptitle("Estimated Noisy States Vs True Noisy Dynamics")
            
            fig2,axs2 = plt.subplots(5,1, figsize=(10,10))
            axs2[0].plot(tvector[:len(y_hat)], y_hat[:,0], label=r'$\text{Azimuth} a \to g$ Estimated')
            # axs2[0].plot(tvector, ydata[0,:],'r--', label="True")
            axs2[0].plot(tvector, y_hat[:,0],'r--', label=r'$\text{Azimuth} a \to g$ True')
            #axs2[0].plot(tvector[:len(y_nonlinear)], nonlinear_azimuth_bounded,'g--', label="Nonlinear Model")
            axs2[0].set_ylabel(r'$\text{Azimuth} a \to g$')
            axs2[0].set_xlabel('Time [s]')
            axs2[0].legend()
            axs2[1].plot(tvector[:len(y_hat)],y_hat[:,1],label = "Estimated")
            # axs2[1].plot(tvector,ydata[1,:],'r--', label = "True")
            axs2[1].plot(tvector,y_hat[:,1],'r--', label = "True")
            #axs2[1].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,1],'g--', label="Nonlinear Model")
            axs2[1].set_ylabel(r'range')
            axs2[1].set_xlabel('Time [s]')
            axs2[1].legend()
            axs2[2].plot(tvector[:len(y_hat)],y_hat[:,2], label = "Estimated")
            # axs2[2].plot(tvector,ydata[2,:],'r--', label = "True")
            axs2[2].plot(tvector,y_hat[:,2],'r--', label = "True")
            #axs2[2].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,2],'g--', label="Nonlinear Model")
            axs2[2].set_ylabel(r'$\text{bearing } g \to a$')
            axs2[2].set_xlabel('Time [s]')
            axs2[2].legend()
            axs2[3].plot(tvector[:len(y_hat)],y_hat[:,3], label = "Estimated")
            # axs2[3].plot(tvector,ydata[3,:],'r--', label = "True")
            axs2[3].plot(tvector,y_hat[:,3],'r--', label = "True")
            #axs2[3].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,3],'g--', label="Nonlinear Model")
            axs2[3].set_ylabel(r'UAV $\xi$ (GPS)')
            axs2[3].set_xlabel('Time [s]')
            axs2[3].legend()
            axs2[4].plot(tvector[:len(y_hat)],y_hat[:,4], label = "Estimated")
            # axs2[4].plot(tvector,ydata[4,:],'r--', label = "True")
            axs2[4].plot(tvector,y_hat[:,4],'r--', label = "True")
            #axs2[4].plot(tvector[:len(y_nonlinear)], y_nonlinear[:,4],'g--', label="Nonlinear Model")
            axs2[4].set_ylabel(r'UAV $\eta$ (GPS)')
            axs2[4].set_xlabel('Time [s]')
            axs2[4].legend()
            fig2.suptitle("Approximate Linearized Model Data Vs True Measurement Data")

            upper_bounds = 2*sigma
            lower_bounds = -2*sigma
            fig3,axs3 = plt.subplots(6,1, figsize=(10,10))
            axs3[0].plot(t,epsilon[:time_steps,0], label="Error")
            axs3[0].plot(t,upper_bounds[:time_steps,0],'r--', label="upper Bound")
            axs3[0].plot(t,lower_bounds[:time_steps,0],'r--', label="Lower Bound")
            axs3[0].set_ylabel(r'$\xi_g$ [m]')
            axs3[0].set_xlabel('Time [s]')
            axs3[0].legend()
            axs3[1].plot(t,epsilon[:time_steps,1], label="Error")
            axs3[1].plot(t,upper_bounds[:time_steps,1],'r--', label="upper Bound")
            axs3[1].plot(t,lower_bounds[:time_steps,1],'r--', label="Lower Bound")
            axs3[1].set_ylabel(r'$\eta_g$ [m]')
            axs3[1].set_xlabel('Time [s]')
            axs3[1].legend()
            axs3[2].plot(t,epsilon[:time_steps,2], label="Error")
            axs3[2].plot(t,upper_bounds[:time_steps,2],'r--', label="upper Bound")
            axs3[2].plot(t,lower_bounds[:time_steps,2],'r--', label="Lower Bound")
            axs3[2].set_ylabel(r'$\theta_g$ [m]')
            axs3[2].set_xlabel('Time [s]')
            axs3[2].legend()
            axs3[3].plot(t,epsilon[:time_steps,3], label="Error")
            axs3[3].plot(t,upper_bounds[:time_steps,3],'r--', label="upper Bound")
            axs3[3].plot(t,lower_bounds[:time_steps,3],'r--', label="Lower Bound")
            axs3[3].set_ylabel(r'$\xi_a$ [m]')
            axs3[3].set_xlabel('Time [s]')
            axs3[3].legend()
            axs3[4].plot(t,epsilon[:time_steps,4], label="Error")
            axs3[4].plot(t,upper_bounds[:time_steps,4],'r--', label="upper Bound")
            axs3[4].plot(t,lower_bounds[:time_steps,4],'r--', label="Lower Bound")
            axs3[4].set_ylabel(r'$\eta_a$ [m]')
            axs3[4].set_xlabel('Time [s]')
            axs3[4].legend()
            axs3[5].plot(t,epsilon[:time_steps,5], label="Error")
            axs3[5].plot(t,upper_bounds[:time_steps,5],'r--', label="upper Bound")
            axs3[5].plot(t,lower_bounds[:time_steps,5],'r--', label="Lower Bound")
            axs3[5].set_ylabel(r'$\theta_a$ [rad]')
            axs3[5].set_xlabel('Time [s]')
            axs3[5].legend()
            fig3.suptitle("State Errors and 2Sigma Bounds")

            plt.show()

        case "UKF":
            #Run UKF SIM
            pass
