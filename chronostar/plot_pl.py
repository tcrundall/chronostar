import pickle
import matplotlib.pyplot as plt

(stars,times,xyzuvw,xyzuvw_cov)=pickle.load(open("traceback_save.pkl"))

pl_stars = []
for i in range(len(stars)):
    X = xyzuvw[i,80,0]
    Y = xyzuvw[i,80,1]
    Z = xyzuvw[i,80,2]
    if (-1900<X and X<-1540) and (2000<Y and Y<2600) and (Z<-20 and Z>-45):
        pl_stars.append(i)
        

for i in range(len(stars)):
    if i in pl_stars:
        colour = 'r'
    else:
        colour = 'b'
    plt.plot(xyzuvw[i,:,0], xyzuvw[i,:,1], colour)

plt.scatter(xyzuvw[:,0,0], xyzuvw[:,0,1])

plt.title("Traceback of bright stars near Pleiades")
plt.xlabel("X [pc]")
plt.ylabel("Y [pc]")
plt.savefig("plots/pleiades_traced_XY.png")
plt.show()

plt.clf()

