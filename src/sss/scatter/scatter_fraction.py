import math
import numpy as np
from scipy import integrate
from .kn_formula import get_scatter_cos_theta,fkn
from ..preprocess import pre_all_scatter_position,pre_atten,pre_lors_atten,pre_sumup_of_emission,get_all_crystal_position,get_image
# from srf.external.stir.function import get_scanner
from ..io import get_data
from scipy.special import ndtr
from numba import jit,cuda
import srfnef as nef
import time
import sys

@jit
def scatter_fraction(config,index,lors,scanner):
    emission_image = get_image(config['emission'])
    u_map = get_image(config['transmission'])
    scatter_position = pre_all_scatter_position(emission_image)
    crystal_position = get_all_crystal_position(scanner)
    sumup_emission = pre_sumup_of_emission(emission_image,crystal_position,scatter_position,config)
    atten = pre_atten(u_map,crystal_position,scatter_position,config)  
    atten_lors = pre_lors_atten(u_map,crystal_position[lors[index,0]],crystal_position[lors[index,1]],config).reshape(-1,1)  
    scatter = np.zeros((index.size,1),dtype=np.float32)
    scale = np.zeros((index.size,1),dtype=np.float32)
    lors_part = lors[index,:]
    loop_all_lors[(512,512),(16,16)](scanner.nb_detectors_per_ring,scanner.nb_blocks_per_ring,np.array(scanner.blocks.shape,dtype=np.int32),
                    scatter_position,crystal_position,config['energy']['window'][0],config['energy']['window'][1],math.sqrt(511)*config['energy']['resolution'],
                    atten,sumup_emission,scatter_position.shape[0],u_map.data,np.array(u_map.size,dtype=np.float32),np.array(u_map.data.shape,dtype=np.int32),
                    lors_part,scatter,scale)
    efficiency_without_scatter = eff_without_scatter(config['energy']['window'][0],config['energy']['window'][1],
                    math.sqrt(511)*config['energy']['resolution'])*5/(1022*math.pi)**0.5/config['energy']['resolution']
    scatter = scatter*efficiency_without_scatter*(scanner.blocks.size[1]*scanner.blocks.size[2]/scanner.blocks.shape[1]/scanner.blocks.shape[2])**2*5/(1022*math.pi)**0.5/config['energy']['resolution']/4/math.pi
    return scatter,scale,atten_lors


@cuda.jit
def loop_all_lors(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,scatter_position,crystal_position,low_energy,high_energy,
                    resolution,atten,sumup_emission,num_scatter,u_map,size,shape,lors,scatter_ab,scale_ab):
    c, j = cuda.grid(2)
    i = 512 * 16 * c + j
    if i <scatter_ab.shape[0]:
        a = int(lors[i,0])
        b = int(lors[i,1])
        scatter_ab[i,0] = loop_all_s(scatter_position,crystal_position[a,:],crystal_position[b,:],low_energy,high_energy,resolution,
            sumup_emission[a*num_scatter:(a+1)*num_scatter],sumup_emission[b*num_scatter:(b+1)*num_scatter],atten[a*num_scatter:(a+1)*num_scatter],
            atten[b*num_scatter:(b+1)*num_scatter],nb_detectors_per_ring,nb_blocks_per_ring,grid_block,u_map,size,shape)
        scale_ab[i,0] = get_scale(crystal_position[a,:],crystal_position[b,:],nb_detectors_per_ring,nb_blocks_per_ring,grid_block)

@cuda.jit(device=True)
def get_scale(A,B,nb_detectors_per_ring,nb_blocks_per_ring,grid_block):
    area = project_area(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,A,B)
    return (distance_a2b(A[0],A[1],A[2],B[0],B[1],B[2])/area)**2


@cuda.jit(device=True)
def loop_all_s(scatter_position,A,B,low_energy,high_energy,resolution,sumup_emission_s2a,sumup_emission_s2b,atten_s2a,atten_s2b,
                nb_detectors_per_ring,nb_blocks_per_ring,grid_block,u_map,size,shape):    
    scatter_ab = 0
    for i in range(int(scatter_position.shape[0])):
        S = scatter_position[i,:]
        cos_theta = get_scatter_cos_theta(A,S,B)
        scattered_energy = 511/(2-cos_theta)
        Ia = atten_s2a[i]*math.exp(math.log(atten_s2b[i])/scattered_energy*511)*sumup_emission_s2a[i]
        Ib = math.exp(math.log(atten_s2a[i])/scattered_energy*511)*atten_s2b[i]*sumup_emission_s2b[i]
        scatter_ab += (project_area(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,S,A)*eff_with_scatter(low_energy,high_energy,scattered_energy,resolution)*
            project_area(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,S,B)*fkn(A,S,B,u_map,size,shape)*(Ia+Ib)/
            (distance_a2b(S[0],S[1],S[2],A[0],A[1],A[2]))**2/(distance_a2b(S[0],S[1],S[2],B[0],B[1],B[2]))**2)   
    return scatter_ab

@jit(nopython=True)
def eff_without_scatter(low_energy_window,high_energy_window,energy_resolution):
    """
    calulate detection efficiency according to lors energy with no scatter
    """
    eff = 0
    for i in range(low_energy_window,high_energy_window,5):
        eff += math.exp(-(float(i)-511)**2/2/energy_resolution**2)
    return eff

@cuda.jit(device=True)
def eff_with_scatter(low_energy_window,high_energy_window,scattered_energy,energy_resolution):
    """
    calulate detection efficiency according to lors energy with scatter
    """      
    eff = 0
    for i in range(low_energy_window,high_energy_window,5):
        eff += math.exp(-(float(i)-scattered_energy)**2/2/energy_resolution**2)
    return eff
    

@cuda.jit(device=True)
def project_area(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,pa,pb):
    """
    calculate LOR ab projection area on pb,which is crystal area*cos(theta)
    """   
    theta_normal = get_block_theta(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,pb)
    theta = (((pb[0]-pa[0])*math.cos(theta_normal)+(pb[1]-pa[1])*math.sin(theta_normal))
                    /distance_a2b(pa[0],pa[1],pa[2],pb[0],pb[1],pb[2])) 
    return theta

@cuda.jit(device=True)
def get_block_theta(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,p):
    event_norm = distance_a2b(p[0],p[1],p[2],0,0,0)
    before_theta = p[0]/event_norm
    theta_event = math.acos(before_theta)/math.pi*180
    if p[1]<0:
        theta_event = 360-theta_event
    fixed_theta = (theta_event+180/nb_detectors_per_ring*grid_block[1])%360
    id_block = math.floor(fixed_theta/360*nb_blocks_per_ring)
    return id_block/nb_blocks_per_ring*2*math.pi

@cuda.jit(device=True)
def distance_a2b(x1,y1,z1,x2,y2,z2):
    return ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5

def get_scanner(config):
    block = nef.Block(np.array(config['block']['size']),
                  np.array(config['block']['grid']))
    return nef.PETCylindricalScanner(config['ring']['inner_radius'],
                        config['ring']['outer_radius'],
                        config['ring']['nb_rings'],
                        config['ring']['nb_blocks_per_ring'],
                        config['ring']['gap'],
                        block)