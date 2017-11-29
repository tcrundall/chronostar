from __future__ import division, print_function
from vpython import *
import numpy as np
scene = canvas(title='3D scene')

simulating = True
# make workaround axes
axs = 5
xline = box(pos=vec(axs/2,0,0), size=vec(axs,0.1,0.1), color=color.red)
yline = box(pos=vec(0,axs/2,0), size=vec(0.1,axs,0.1), color=color.blue)
zline = box(pos=vec(0,0,axs/2), size=vec(0.1,0.1,axs), color=color.cyan)
debug = False
if debug:
    # desired shape of ellipsoid
    majoraxis = vec(1/2, -1/np.sqrt(2), 1/2)
    minoraxis = vec(1/2, 1/np.sqrt(2), 1/2)
    # otheraxis = vec(1/np.sqrt(2), 0, -1/np.sqrt(2))
    print(cross(majoraxis, minoraxis))
    print(dot(majoraxis, minoraxis))
    
scene.background = color.white

majoraxis = vec(4, -2, 1)
minoraxis = vec(1, 1, -2)
#majoraxis = vec(4, 0, 0)
#minoraxis = vec(0, 2, 0)

cros_prod = cross(majoraxis, minoraxis)
otheraxis = cros_prod / mag(cros_prod)

# insert pointers for axes
la = arrow(axis=majoraxis, shaftwidth=0.05)
ha = arrow(axis=minoraxis, shaftwidth=0.05)
wa = arrow(axis=otheraxis, shaftwidth=0.05)

margin = 0.1
scale = 0.5
L = mag(majoraxis)
H = mag(minoraxis)
W = mag(otheraxis)

star = ellipsoid(length=L, height=H, width=W, axis=majoraxis,
        up = minoraxis, opacity=0.2, color=color.red, make_trail=True)

if simulating:
    # time
    star.velocity = vec(1,1,0)
    t = 0
    deltat = 0.005
    maxt = 10
    #omega = 1.0 / maxt  # 1 revoluion per run

    while t < maxt:
        rate(100)
        star.pos = star.pos + star.velocity*deltat
        #star.rotate(angle=2*pi*omega*deltat)
        star.length = star.length + 3*deltat
        for obj in scene.objects:
            if obj.__class__ == arrow:
                obj.pos = star.pos
        la.length = star.length * scale + margin
        ha.length = star.height * scale + margin
        wa.length = star.width * scale  + margin
        t = t + deltat

#for i in range(-5,4
#    sphere(pos=vec(2*i, 0, 0))
#    ellipsoid(pos=vec(2*i, 0, 2), length=L, height=H, width=W,
#            axis=axis, make_trail=True)

#for obj in scene.objects:
#    if obj.__class__ == sphere:
#        obj.color = color.red
#    if obj.__class__ == ellipsoid:
#        obj.color = color.yellow
#        obj.velocity = vector(1,0,0)
        
# mybox = box(pos=vec(0,1,0), size=vec(2,2,2))
# lamp = local_light(pos=vec(5,5,2), color=color.yellow)

