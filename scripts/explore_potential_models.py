from galpy.potential import PowerSphericalPotentialwCutoff, MiyamotoNagaiPotential, NFWPotential, verticalfreq, MWPotential2014

scale_height_factor = [0.5, 1.0, 2.0]
default_vper = 1./verticalfreq(MWPotential2014, 1.0)

for shf in scale_height_factor:
    bp= PowerSphericalPotentialwCutoff(alpha=1.8,rc=1.9/8.,normalize=0.05)
    mp= MiyamotoNagaiPotential(a=3./8.,b=shf*0.28/8.,
                               normalize=.6)
    np= NFWPotential(a=16/8.,normalize=.35)

    my_mwpotential2014 = [bp,mp,np]

    vfreq = verticalfreq(my_mwpotential2014, 1.0)
    vper = 1./vfreq

    if shf == 1.0:
        assert verticalfreq(my_mwpotential2014, 1.0) ==\
               verticalfreq(MWPotential2014, 1.0)

    print('Scale height factor:        {}'.format(shf))
    print('Vertical freq:              {:.2f}'.format(verticalfreq(my_mwpotential2014, 1.0)))
    print('normalised vertical period: {:.2f}'.format(vper/default_vper))