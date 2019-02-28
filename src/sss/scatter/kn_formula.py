from numba import jit
import math

@jit(nopython=True)
def fkn(A,S,B,u_map,size,shape,Z):
    """
    Differential cross section:A to S, and scattering to B
    """
    cos_asb = get_scatter_cos_theta(A,S,B)
    grid_S0 = int(S[0]*shape[0]/size[0])
    grid_S1 = int(S[1]*shape[1]/size[1])
    grid_S2 = int(S[2]*shape[2]/size[2])
    differential_value = (1/(2-cos_asb)**2*((1+cos_asb**2)/2
                          +(1-cos_asb)**2/2/(2-cos_asb)**2))
    sigma = math.pi*Z*(52/9-3*math.log(3))
    return differential_value/sigma*u_map[grid_S0,grid_S1,grid_S2]

@jit(nopython=True)
def get_scatter_cos_theta(A,S,B):
    dis_ab = distance_a2b(A[0],A[1],A[2],B[0],B[1],B[2])
    dis_as = distance_a2b(A[0],A[1],A[2],S[0],S[1],S[2])
    dis_sb = distance_a2b(S[0],S[1],S[2],B[0],B[1],B[2])
    return -(dis_as**2+dis_sb**2-dis_ab**2)/(2*dis_as*dis_sb)

@jit(nopython=True)
def distance_a2b(x1,y1,z1,x2,y2,z2):
    return ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5