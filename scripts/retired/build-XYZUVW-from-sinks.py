import astropy.units as u
import numpy as np

if __name__ == '__main__':
    filename = '../data/reduced_sinks_evol.dat'
    savefile = '../data/sink_init_xyzuvw.npy'
    raw_dict = {}
    with open(filename, 'r') as fp:
        hline = fp.readline().split()
        data = np.loadtxt(fp)
        for i, cname in enumerate(hline):
            raw_dict[cname[4:]] = data[:,i]

    # Convert desired values from cgs units to pc, km/s, Myr
    
    XYZUVW_dict = {}
    XYZUVW_dict['part_tag'] = raw_dict['part_tag']

    # check no duplicates
    assert len(XYZUVW_dict['part_tag']) == len(set(XYZUVW_dict['part_tag']))
    nstars = len(XYZUVW_dict['part_tag'])
    XYZUVW_arr = np.zeros((0,nstars))

    my_pos_labels = 'XYZ'
    his_pos_labels = ['posx', 'posy', 'posz']
    for my_pos, his_pos in zip(my_pos_labels, his_pos_labels):
        XYZUVW_dict[my_pos] = (raw_dict[his_pos] * u.cm).to('pc').value
        XYZUVW_arr = np.vstack((XYZUVW_arr, XYZUVW_dict[my_pos]))


    my_vel_labels = 'UVW'
    his_vel_labels = ['velx', 'vely', 'velz']
    for my_vel, his_vel in zip(my_vel_labels, his_vel_labels):
        XYZUVW_dict[my_vel] = (raw_dict[his_vel] * u.cm / u.s).\
            to('km/s').value
        XYZUVW_arr = np.vstack((XYZUVW_arr, XYZUVW_dict[my_vel]))

    XYZUVW_arr = XYZUVW_arr.T
    np.save(savefile, XYZUVW_arr)

