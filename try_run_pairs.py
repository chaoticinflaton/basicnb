# -*- coding: utf-8 -*-
"""

Basic N-Body Simulator

Creates a gif in the nbs folder at the current location. 

Please create folder beforehand. 

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

g = 0.05 # Gravitational force strength
dt = 0.01 # Timestep

l = 1.0 # Box side. 

soft_len = 0.01 # Softening factor

npart = 100 # No. of particles
nsteps = 500 # No of timesteps

# Generate random positions

a = np.random.rand(npart,3)

x = a[:,0]
x = np.reshape(x,(len(x),1))

y = a[:,1]
y = np.reshape(y,(len(y),1))

z = a[:,2]
z = np.reshape(z,(len(z),1))

# Generate random velocities

v = 0.001*np.random.rand(npart,3)

vx = v[:,0]
vx = np.reshape(vx,(len(x),1))

vy = v[:,1]
vy = np.reshape(vy,(len(x),1))

vz = v[:,2]
vz = np.reshape(vz,(len(x),1))

# Start time computation now. 
# Uses leapfrog integration. 
# First, the new positions are computed using the current velocity.
# Forces are computed at the new positions, and they are used to change the
# velocities. 

for i in range(nsteps):
    
    print i
    
    # Update position and impose periodic BCs by wrapping. 
    
    x = (x + vx*dt)%l
    y = (y + vy*dt)%l
    z = (z + vz*dt)%l
    
    # This is a numpy-way to generate an array which gives the pairwise distances 
    # dx(i,j) = (x_i - x_j)

    dx = x - x.transpose()
    dy = y - y.transpose()
    dz = z - z.transpose()
    
    
    # Denominator for force expression
    mod = np.power(dx*dx + dy*dy + dz*dz + soft_len,1.5)
    
    # Vector components of force
    fx = dx/mod
    fy = dy/mod
    fz = dz/mod
    
    # Summed over all particles, for each particle
    total_f_x = g*np.sum(fx,axis=0)
    total_f_y = g*np.sum(fy,axis=0)
    total_f_z = g*np.sum(fz,axis=0)
    
    total_f_x = np.reshape(total_f_x,(len(total_f_x),1))
    total_f_y = np.reshape(total_f_y,(len(total_f_x),1))
    total_f_z = np.reshape(total_f_z,(len(total_f_x),1))
    
    # Updating velocities using force calculated at NEW positions. 
    vx = vx + total_f_x*dt
    vy = vy + total_f_y*dt
    vz = vz + total_f_z*dt
    
    # Plotting 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x,y,z)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)

    plt.savefig('./nbs/' + str(i).zfill(5) + '.png')
    plt.close()
    
del x,y,z,vx,vy,vz,total_f_x,total_f_y,total_f_z,dx,dy,dz # Free up space

# Create gif.
import imageio
images = []
for i in range(nsteps):
    images.append(imageio.imread('./nbs/' + str(i).zfill(5) + '.png'))
    
imageio.mimsave('./nbs/nb.gif', images,fps=8)
