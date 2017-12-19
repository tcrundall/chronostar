#! /usr/bin/env python
"""run:
    ipython
    >> %run read_in_memb_probs.py
    >> play around :)
"""

import pickle

def calc_average_dpos(cov):
    return (cov[0,0]**0.5 + cov[1,1]**0.5 + cov[2,2]**0.5) / 3

def calc_average_dvel(cov):
    return (cov[3,3]**0.5 + cov[4,4]**0.5 + cov[5,5]**0.5) / 3

try:
    infile = "../results/membership_probs_from_rave.pkl"
    with open(infile, 'r') as fp:
        group_names, gaia_ids, memb_probs, xyzuvws, xyzuvw_covs, sub_tables,\
        best_membs_ixs, all_memb_probs, orig_table = pickle.load(fp)
except IOError:
    infile = "membership_probs_from_rave.pkl"
    with open(infile, 'r') as fp:
        group_names, gaia_ids, memb_probs, xyzuvws, xyzuvw_covs, sub_tables,\
        best_membs_ixs, all_memb_probs, orig_table = pickle.load(fp)

for i, group in enumerate(group_names[:-1]):
    with open("{}_details.csv".format(group), 'w') as fp:
        fp.write("Probabilities, average dPos, average dV, gaia_id,"
                 "RA (degrees), DEC (degrees)\n")
        for j in range(10):
            fp.write("{}, {}, {}, {}, {}, {}\n".format(
                memb_probs[group][j],
                calc_average_dpos(xyzuvw_covs[group][j]),
                calc_average_dvel(xyzuvw_covs[group][j]),
                gaia_ids[group][j],
                sub_tables[group]['RAdeg'][j],
                sub_tables[group]['DEdeg'][j],
            ))

print("Pars now in global space:\ngroup_names\ngaia_ids\n"
      "memb_probs\nxyzuvws\nxyzuvw_covs\nsub_tables\nbest_membs_ixs\n"
      "all_memb_probs\norig_table")

print("All are dictionaries with the keys listed in `group_names`:\n{}"\
    .format(group_names))
