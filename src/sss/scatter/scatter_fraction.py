import math
import numpy as np
from scipy import integrate
from .kn_formula import get_scatter_cos_theta,fkn
from ..preprocess import pre_all_scatter_position,pre_atten,pre_sumup_of_emission,get_all_crystal_position,get_image
# from srf.external.stir.function import get_scanner
from ..io import get_data
from scipy.special import ndtr
from numba import jit,cuda
import srfnef as nef
import time

@jit
def scatter_fraction(config,sinogram,lors,scanner):
    emission_image = get_image(config['emission'])
    u_map = get_image(config['transmission'])
    scatter_position = pre_all_scatter_position(emission_image)
    crystal_position = get_all_crystal_position(scanner)
    sumup_emission = pre_sumup_of_emission(emission_image,crystal_position,scatter_position)
    atten = pre_atten(u_map,crystal_position,scatter_position)  
    index = np.where(sinogram>0)[0].astype(np.int32)
    scatter = np.zeros((index.size,1),dtype=np.float32)
    lors_part = lors[index,:]
    loop_all_lors[(512,512),(16,16)](scanner.nb_detectors_per_ring,scanner.nb_blocks_per_ring,np.array(scanner.blocks.shape,dtype=np.int32),
                    np.array(scanner.blocks.size,dtype=np.int32),scatter_position,crystal_position,config['energy']['window'][0],
                    config['energy']['window'][1],config['energy']['resolution'],atten,sumup_emission,scatter_position.shape[0],
                    u_map.data,np.array(u_map.size,dtype=np.float32),np.array(u_map.data.shape,dtype=np.float32),lors_part,scatter)
    print(np.max(scatter))
    print(np.mean(scatter))
    total_scatter = np.zeros((int(scanner.nb_detectors*(scanner.nb_detectors-1)/2),1),dtype=np.float32)
    total_scatter[index] = scatter
    return total_scatter*0.3/np.max(total_scatter)

# @cuda.jit
# def loop_all_lors(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,size_block,scatter_position,crystal_position,low_energy,high_energy,resolution,atten,sumup_emission,num_scatter,u_map,size,shape,lors,scatter_ab):
#     # scatter_ab = np.zeros((int(nb_detectors*(nb_detectors-1)/2),1))
#     i = cuda.grid(1)
#     if i <scatter_ab.shape[0]:
#         a = int(lors[i,0])
#         b = int(lors[i,1])
#         scatter_ab[i,0] = loop_all_s(scatter_position,crystal_position[a,:],crystal_position[b,:],low_energy,high_energy,resolution,sumup_emission[a*num_scatter:(a+1)*num_scatter],
#         sumup_emission[b*num_scatter:(b+1)*num_scatter],atten[a*num_scatter:(a+1)*num_scatter],atten[b*num_scatter:(b+1)*num_scatter],nb_detectors_per_ring,nb_blocks_per_ring,grid_block,size_block,u_map,size,shape)
#     # return scatter_ab
@cuda.jit
def loop_all_lors(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,size_block,scatter_position,crystal_position,low_energy,high_energy,resolution,atten,sumup_emission,num_scatter,u_map,size,shape,lors,scatter_ab):
    # scatter_ab = np.zeros((int(nb_detectors*(nb_detectors-1)/2),1))
    c, j = cuda.grid(2)
    i = 512 * 16 * c + j
    if i <scatter_ab.shape[0]:
        a = int(lors[i,0])
        b = int(lors[i,1])
        scatter_ab[i,0] = loop_all_s(scatter_position,crystal_position[a,:],crystal_position[b,:],low_energy,high_energy,resolution,sumup_emission[a*num_scatter:(a+1)*num_scatter],
        sumup_emission[b*num_scatter:(b+1)*num_scatter],atten[a*num_scatter:(a+1)*num_scatter],atten[b*num_scatter:(b+1)*num_scatter],nb_detectors_per_ring,nb_blocks_per_ring,grid_block,size_block,u_map,size,shape)


@cuda.jit(device=True)
def loop_all_s(scatter_position,A,B,low_energy,high_energy,resolution,sumup_emission_s2a,sumup_emission_s2b,atten_s2a,atten_s2b,nb_detectors_per_ring,nb_blocks_per_ring,grid_block,size_block,u_map,size,shape):    
    scatter_ab = 0
    efficiency_without_scatter = eff_without_scatter(low_energy,high_energy,resolution)
    for i in range(scatter_position.shape[0]):
        S = scatter_position[i,:]
        cos_theta = get_scatter_cos_theta(A,S,B)
        scattered_energy = 511/(2-cos_theta)
        I = (efficiency_without_scatter*eff_with_scatter(low_energy,high_energy,scattered_energy,resolution)*
            atten_s2a[i]*atten_s2b[i]*scattered_energy/511*(sumup_emission_s2a[i]+sumup_emission_s2b[i]))
        # Ib = (efficiency_without_scatter*eff_with_scatter(low_energy,high_energy,scattered_energy,resolution)*
        #               atten_s2a[i]*atten_s2b[i]*scattered_energy/511*sumup_emission_s2b[i])
        scatter_ab += (project_area(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,size_block,S,A)*project_area(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,size_block,S,B)
                      *fkn(A,S,B,u_map,size,shape,10.0)*I/4/math.pi/(distance_a2b(S[0],S[1],S[2],A[0],A[1],A[2]))**2/(distance_a2b(S[0],S[1],S[2],B[0],B[1],B[2]))**2)   
    return scatter_ab

@cuda.jit(device=True)
def eff_without_scatter(low_energy_window,high_energy_window,energy_resolution):
    """
    calulate detection efficiency according to lors energy with no scatter
    """
    eff = 0
    for i in range(low_energy_window,high_energy_window,5):
        eff += math.exp(-(float(i)-511)**2/2/energy_resolution**2)
    return eff*5/(2*math.pi)**0.5/energy_resolution

@cuda.jit(device=True)
def eff_with_scatter(low_energy_window,high_energy_window,scattered_energy,energy_resolution):
    """
    calulate detection efficiency according to lors energy with scatter
    """      
    eff = 0
    for i in range(low_energy_window,high_energy_window,5):
        eff += math.exp(-(float(i)-scattered_energy)**2/2/energy_resolution**2)
    return eff*5/(2*math.pi)**0.5/energy_resolution
    

@cuda.jit(device=True)
def project_area(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,size_block,pa,pb):
    """
    calculate LOR ab projection area on pb,which is crystal area*cos(theta)
    """   
    theta_normal = get_block_theta(nb_detectors_per_ring,nb_blocks_per_ring,grid_block,pb)
    # theta = -get_scatter_cos_theta(pa,pb,(pb-np.array([math.cos(theta_normal),math.sin(theta_normal),0],dtype=np.float32)))
    theta = (((pb[0]-pa[0])*math.cos(theta_normal)+(pb[1]-pa[1])*math.sin(theta_normal))
                    /distance_a2b(pa[0],pa[1],pa[2],pb[0],pb[1],pb[2])) 
    return size_block[1]*size_block[2]/grid_block[1]/grid_block[2]*theta

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