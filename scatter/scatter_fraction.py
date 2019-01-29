import math
import numpy as np
from scipy import integrate
def scatter_fraction():
    pass

def s2a():
    pass

def atten_s2a():
    pass

def eff_without_scatter(low_energy_window,high_energy_window,energy_resolution):
    """
    calulate detection efficiency according to lors energy with no scatter
    """
    func = lambda a:math.exp(-(a-511)**2/2/energy_resolution**2)
    integral_value = integrate.nquad(func,low_energy_window,high_energy_window)
    return integral_value/(math.sqrt(2*math.pi)*energy_resolution)

def eff_with_scatter(low_energy_window,high_energy_window,scatter_theta,energy_resolution):
    """
    calulate detection efficiency according to lors energy with scatter
    """  
    scattered_energy = 511/(2-math.cos(theta))
    func = lambda a:math.exp(-(a-scattered_energy)**2/2/energy_resolution**2)
    integral_value = integrate.nquad(func,low_energy_window,high_energy_window)
    return integral_value/(math.sqrt(2*math.pi)*energy_resolution)
    
 
def project_area(scanner,pa,pb):
    """
    calculate LOR ab projection area on pb,which is crystal area*cos(theta)
    """   
    crystal_size = scanner.block_size/scanner.block_grid
    theta_normal = get_block_theta(scanner,pb)
    theta = math.acos((pb[0]-pa[0],pb[1]-pa[1],pb[2]-pa[2])@(cos(theta_normal),sin(theta_normal),0)
                    /distance_a2b(pa[0],pa[1],pa[2],pb[0],pb[1],pb[2])) 
    return crystal_size[1]*crystal_size[2]*theta

def get_block_theta(scanner,p):
    event_norm = distance_a2b(p[0],p[1],p[2],0,0,0)
    dot_mul = p[:2]@np.array([[1],[0]])
    before_theta = dot_mul/event_norm.reshape(-1,1)
    theta_event = math.degrees(math.acos(before_theta))
    if p[1]<0:
        theta_event = 360-theta_event
    fixed_theta = (theta_event+180/scanner.nb_detectors_per_ring*scanner.blocks[0].grid[1])%360
    id_block = math.floor(fixed_theta/360*num_block)
    return id_block/scanner.nb_blocks_per_ring*2*math.pi

def distance_a2b(x1,y1,z1,x2,y2,z2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)

def sumup_of_events(image,pa,pb):
    projection
