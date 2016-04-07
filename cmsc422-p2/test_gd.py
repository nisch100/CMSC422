import gd
import mlGraphics
from numpy import *
from pylab import *

#print gd.gd(lambda x: x**2, lambda x: 2*x, 10, 10, 0.2)

#step_size = .02
#num_iter = 100
#startx = 10.5
#print "Step size: " + str(step_size) + " Starting position: " + str(startx)
#
#x, trajectory = gd.gd(lambda x: (x-10)**4 + (x-10)**3 - (x-10)**2 - (x-10) + 5 , lambda x: 4*(x**3) - 117*(x**2) + 1138*x -3681, startx, num_iter, step_size)
#
##x, trajectory = gd.gd(lambda x: linalg.norm(x)**2, lambda x: 2*x, array([10,5]), 100, 0.2)
#print x
#
#plot(trajectory)
#title('step size: ' + str(step_size))
#show()
#exit

x = [0.1, 0.15, 0.2, 6, 6.5, 7]

for step_size in x:
    print "Step size: " + str(step_size)
    x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, step_size)
    print x
    plot(trajectory)
    title('step size: ' + str(step_size))
#    show()
#    savefig('gd-'+str(step_size)+'-step.png')
#    show()

