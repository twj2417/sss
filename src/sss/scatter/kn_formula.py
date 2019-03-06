from numba import jit,cuda
import math

@cuda.jit(device=True)
def fkn(A,S,B,u_map,size,shape,Z):
    """
    Differential cross section:A to S, and scattering to B
    """
    cos_asb = get_scatter_cos_theta(A,S,B)
    grid_S0 = int((S[0]/size[0]+0.5)*shape[0])
    grid_S1 = int((S[1]/size[1]+0.5)*shape[1])
    grid_S2 = int((S[2]/size[2]+0.5)*shape[2])
    differential_value = (1.0/(2.0-cos_asb)**2*((1.0+cos_asb**2)/2.0
                          +(1.0-cos_asb)**2/2.0/(2.0-cos_asb)))
    sigma = math.pi*Z*(52.0/9.0-3.0*math.log(3.0))
    return differential_value/sigma*u_map[grid_S0,grid_S1,grid_S2]

@cuda.jit(device=True)
def get_scatter_cos_theta(A,S,B):
    dis_ab = distance_a2b(A[0],A[1],A[2],B[0],B[1],B[2])
    dis_as = distance_a2b(A[0],A[1],A[2],S[0],S[1],S[2])
    dis_sb = distance_a2b(S[0],S[1],S[2],B[0],B[1],B[2])
    return -(dis_as**2+dis_sb**2-dis_ab**2)/(2*dis_as*dis_sb)

@cuda.jit(device=True)
def distance_a2b(x1,y1,z1,x2,y2,z2):
    return ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5