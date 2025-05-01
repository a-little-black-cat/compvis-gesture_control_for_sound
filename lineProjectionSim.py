from math import sqrt
import matplotlib.pyplot as plt
#From the video:
#https://www.youtube.com/watch?v=iMrRwXtab6Y&list=PLhwIOYE-ldwL6h-peJADfNm8bbO3GlKEy&index=4

# Write some code that simulates the projection of a line
# segment defined by two points (-5,Z) and (5,Z), where Z
# ranges from 10 to 1000, and assuming a camera focal
# length of f=1.
#
# For each distance Z, project the two points into a 1-D
# sensor under perspective projection, and compute the
# length of the segment.
#
# Plot this length as a function of distance Z to see how size
# changes as a function of distance to the camera.


X1 = -5
X2 = 5
f = 1 #focal length

L=[]
for Z in range(10,1000):
    x1 = -f*X1/Z
    x2 = -f*X2/Z
    L.append(sqrt((x1-x2)**2))

plt.plot(L)
plt.xlabel('distance(Z)')
plt.ylabel('length')