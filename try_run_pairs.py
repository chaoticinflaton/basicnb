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

r = np.array([x,y,z])
v = np.array([vx,vy,vz])

# Start time computation now. 
# Uses leapfrog integration. 
# First, the new positions are computed using the current velocity.
# Forces are computed at the new positions, and they are used to change the
# velocities. 

for i in range(nsteps):
    
    print i
    
    # Update position and impose periodic BCs by wrapping. 
    r = (r + v*dt)%l
    
    # Compute pairwise distances in each dimension
    dr_ij = np.array([el - el.transpose() for el in r])
    
    # Calculate |r|^(3/2) for each pair. 
    den = np.power(np.sum(dr_ij**2,axis=0),1.5) + soft_len
    
    # Finally, calculate force but summing over all source particles for
    # each particle. 
    force = g*np.sum(dr_ij / den,axis=1)
    force = np.reshape(force,(3,npart,1))
    
    # Update velocities using new force. 
    v = v + force*dt
    
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
