#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle

def gaussian(x, mu, sig):
    amp = 1 / np.sqrt(2. * np.pi * np.power(sig, 2.))
    return amp * np.exp(-np.power(x-mu, 2.) / (2 * np.power(sig,2.)))

def generate_stars(nstars, pos_mu, pos_sig, vel_mu, vel_sig):
    """
    Generate a group of [nstars] stars with initial positions and velocities
    normally distributed with appropriate means and stdevs.
    Generated stars will have negligible but non-zero position and velocitie
    uncertainties
    """
    init_pos_mu  = np.random.normal(pos_mu, pos_sig, nstars)
    init_vel_mu  = np.random.normal(vel_mu, 1, nstars)
    init_pos_sig = np.zeros(nstars) + 0.01
    init_vel_sig = np.zeros(nstars) + 0.01

    gen_stars = np.vstack([
        init_pos_mu, init_pos_sig, init_vel_mu, init_vel_sig
        ]).T
    return gen_stars


def project_star(pars, time, back=False):
    """
    Purpose: Take a star's currnet position, velocity and uncertainties
             and porject forward in time a certain amount
    Input: [position, sig_pos, velocity, sig_vel]
    Output: [position, sig_pos, velocity, sig_vel]
    """
    if time == 0:
        return pars
    
    if back:
        direction = -1.0
    else:
        direction = 1.0

    x, sig_x, v, sig_v = pars
    sig_v_t = sig_v*time
    #sig_new = sig_x * sig_v_t / (np.sqrt(sig_x**2 + sig_v_t**2))
    
    #The observed position and velocity have independent errors, so the uncertainty
    #at a time in the past is the sum of the variances of the component due to each
    #term.
    sig_new = np.sqrt(sig_x**2 + sig_v_t**2)
    #sig_new = sig_x + sig_v_t
    x_new = x + v * time * direction
    return [x_new, sig_new, v, sig_v]

def project_stars(stars, time, back=False):
    new_stars = np.zeros(np.shape(stars))
    for i in range(np.shape(stars)[0]):
        new_stars[i] = project_star(stars[i],time,back)
    return new_stars

def star_pdf(x, pars):
    return gaussian(x, pars[0], pars[1])

def group_pdf(x, stars, scale=1):
    total = 0
    for star in stars:
        total += gaussian(x, star[0], star[1])
    return scale*total

def get_measurement(pars, pos_acc=0.1, vel_acc=0.2):
    """
    Takes a 'true' star, and returns a measurment with accuracy 0.1 pc
    and 0.2 km/s
    """
    pos = pars[0]
    vel = pars[2]
    measured_pos = np.random.normal(pars[0], pos_acc)
    measured_vel = np.random.normal(pars[2], vel_acc)
    return np.array([measured_pos, pos_acc, measured_vel, vel_acc])

def get_measurements(stars, pos_acc=0.1, vel_acc=0.2):
    measured_stars = np.zeros(stars.shape)
    for i in range(measured_stars.shape[0]):
        measured_stars[i] = get_measurement(stars[i], pos_acc, vel_acc)
    return measured_stars

def get_group_size(stars):
    centre = np.mean(stars[:,0])
    total_dist = np.sum(np.abs(stars[:,0] - centre))/(stars.shape[0])
    return total_dist
    

if __name__ == '__main__':
    np.random.seed(0)
    # generating "true" stars
    nstars = 50
    pos_mu  = 0
    pos_sig = 5
    vel_mu  = 5
    vel_sig = 2

    true_age = 20
    
    init_pos_mu  = np.random.normal(pos_mu, pos_sig, nstars)
    init_vel_mu  = np.random.normal(vel_mu, 1, nstars)
    init_pos_sig = np.zeros(nstars) + 1 #0.01
    init_vel_sig = np.zeros(nstars) + 0.01

    mystars = np.vstack([
        init_pos_mu, init_pos_sig, init_vel_mu, init_vel_sig
        ]).T
    
    max_pos = true_age*(vel_mu+0.5*vel_sig) + pos_mu
    min_pos = pos_mu - 2 * (pos_sig + vel_sig * true_age)
    xs = np.linspace(min_pos,max_pos,10000)
#    for time in range(0,true_age+1,5):
#        #plt.plot(xs, star_pdf(xs, project_star(mystars[0], time)), label=time)
#        plt.plot(xs, group_pdf(xs, project_stars(mystars, time)), label=time)
#    plt.legend(loc='best')
#    plt.show()

    projected_stars = project_stars(mystars, true_age)

    pos_error=0.05
    vel_error=0.05
    measured_stars = get_measurements(projected_stars, pos_error, vel_error)

    n_time_steps = 41
    npars = 4
    trace_back = np.zeros((n_time_steps, nstars, npars))

    times = np.linspace(0, 2*true_age, n_time_steps)
    
    for t_ix in range(n_time_steps):
        trace_back[t_ix] = project_stars(measured_stars, times[t_ix], True)

    for t_ix in range(n_time_steps):
        plt.plot(xs, group_pdf(xs, trace_back[t_ix]), label=t_ix)

#    for time in range(0,true_age+1,5):
#        #plt.plot(xs, star_pdf(xs, project_star(mystars[0], time)), label=time)
#        plt.plot(
#            xs, group_pdf(
#                xs, project_stars(measured_stars, time, True)),
#            label=time)
    plt.show()

    #plt.plot(xs, group_pdf(xs, mystars), label="origin")
    true_age_ix = 20
    plt.plot(xs, group_pdf(xs, trace_back[true_age_ix], scale=0.1), label="traceback")
    for group in trace_back[true_age_ix]:
        plt.plot(xs, star_pdf(xs, group))
    plt.show()

    pickle.dump(
        (trace_back, n_time_steps, nstars, times, mystars),
        open("data.pkl", 'w'))
