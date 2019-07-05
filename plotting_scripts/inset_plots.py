import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])

ax1.plot(range(10), color='red')
ax2.plot(range(6)[::-1], color='green')

#plt.show()
plt.clf()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes



fig2, ax1 = plt.subplots()

ax1.plot(range(10), color='red')

ax_ins = inset_axes(ax1, width='40%', height='30%', loc=2,
                    borderpad=2.,
                    # axes_kwargs={'xmargin':0.}
                    )
ax_ins.plot(range(6)[::-1], color='green')
# ax_ins.plot(range(6)[::-1], color='green')
plt.savefig('dummy_inset.png')
