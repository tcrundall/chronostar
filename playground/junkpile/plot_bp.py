import pickle
import matplotlib.pyplot as plt

(stars,times,xyzuvw,xyzuvw_cov)=pickle.load(open("traceback_save.pkl"))

bp_stars = []
#for i in range(len(stars)):
#    X = xyzuvw[i,80,0]
#    Y = xyzuvw[i,80,1]
#    Z = xyzuvw[i,80,2]
#    if (-1900<X and X<-1540) and (2000<Y and Y<2600) and (Z<-20 and Z>-45):
#        pl_stars.append(i)
        

for i in range(len(stars)):
    if i in bp_stars:
        colour = 'r'
    else:
        colour = 'b'
    plt.plot(xyzuvw[i,:20,1], xyzuvw[i,:20,2])

plt.scatter(xyzuvw[:,0,1], xyzuvw[:,0,2])

plt.title("Orbital traceback of bright stars near Beta Pictoris")
plt.xlabel("Y [pc]")
plt.ylabel("Z [pc]")
plt.savefig("plots/betapic_tracebackYZ.png")
plt.show()

plt.clf()

