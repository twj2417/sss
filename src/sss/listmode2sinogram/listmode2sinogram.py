import numpy as np
import math
import matplotlib.pyplot as plt
import pylab
import gc
from srf.external.stir.function import get_scanner
from srf.io.listmode import load_h5,save_h5
from numba import jit

def norm(x):
    return np.sqrt(np.square(x[:,0])+np.square(x[:,1]))

def get_block_theta(event):
    event_norm = norm(event).reshape(-1,1)
    dot_mul = event[:,:2]@np.array([[1],[0]])
    before_theta = dot_mul/event_norm
    theta = np.arccos(before_theta)
    return np.rad2deg(theta)

def block_id(event,nb_detectors_per_ring,block_grid_y,num_block):
    theta_event = get_block_theta(event)
    index = np.where(event[:,1]<0)[0]
    theta_event[index] = 360*np.ones_like(theta_event[index])-theta_event[index]
    fixed_theta = np.mod(theta_event+180/nb_detectors_per_ring*block_grid_y*np.ones_like(theta_event),360)
    return np.floor(fixed_theta/360*num_block)

def crystal_id(scanner,event):
    id_block = block_id(event,scanner.nb_detectors_per_ring,scanner.blocks[0].grid[1],scanner.nb_blocks_per_ring)
    block_center_x = (scanner.inner_radius+scanner.outer_radius)/2*np.cos(2*np.pi*id_block/scanner.nb_blocks_per_ring)
    block_center_y = (scanner.inner_radius+scanner.outer_radius)/2*np.sin(2*np.pi*id_block/scanner.nb_blocks_per_ring)
    vector_radial = np.hstack((np.cos(2*(np.pi)*id_block/scanner.nb_blocks_per_ring),np.sin(2*(np.pi)*id_block/scanner.nb_blocks_per_ring)))
    vector_tang = np.hstack((np.cos(2*np.pi*id_block/scanner.nb_blocks_per_ring+np.pi/2),np.sin(2*np.pi*id_block/scanner.nb_blocks_per_ring+np.pi/2)))
    dist_vector = np.hstack((block_center_x.reshape(-1,1),block_center_y.reshape(-1,1)))-event[:,:2]
    dist = (dist_vector*vector_radial).sum(axis=1).reshape(-1,1)
    pro_point = np.tile(dist,(1,2))*vector_radial+event[:,:2]
    edge_block = np.hstack((block_center_x.reshape(-1,1),block_center_y.reshape(-1,1)))-vector_tang*scanner.blocks[0].size[1]/2
    id_crystal = np.floor(norm(pro_point-edge_block)/scanner.blocks[0].size[1]*scanner.blocks[0].grid[1])   
    return id_block*scanner.blocks[0].grid[1]+id_crystal.reshape(-1,1)

def ring_id(scanner,event_z):
    return np.floor((event_z+scanner.blocks[0].size[2]*scanner.nb_rings/2*np.ones_like(event_z))/scanner.axial_length*scanner.nb_rings*scanner.blocks[0].grid[1])

def position2detectorid(scanner, event):
    ring_id1 = ring_id(scanner,event[:,2]).reshape(-1,1)
    crystal_id1 = crystal_id(scanner,event[:,:2])
    ring_id2 = ring_id(scanner,event[:,5]).reshape(-1,1)
    crystal_id2 = crystal_id(scanner,event[:,3:5])
    return np.hstack((ring_id1*scanner.nb_detectors_per_ring+crystal_id1,ring_id2*scanner.nb_detectors_per_ring+crystal_id2))

def change_id(data):
    result = np.array(data)
    index = np.where(data[:,0]<data[:,1])[0]
    result[index,0] = data[index,1]
    result[index,1] = data[index,0]
    return result

def detectorid(scanner,events):
    id_event = position2detectorid(scanner,events)
    return change_id(id_event)

def cal_sinogram(list_mode_data,scanner):
    total_num_crystal = scanner.nb_rings*scanner.blocks[0].grid[2]*scanner.nb_detectors_per_ring
    weight = np.zeros((int(total_num_crystal*(total_num_crystal-1)/2),1))
    for i in range(int(list_mode_data.shape[0])):
        pos = int((list_mode_data[i,0]-1)*list_mode_data[i,0]/2+list_mode_data[i,1])
        weight[pos] = weight[pos] + list_mode_data[i,2]
    return weight

@jit(nopython=True)
def get_all_lors_id(total_num_crystal):
    # total_num_crystal = scanner.nb_rings*scanner.blocks[0].grid[2]*scanner.nb_detectors_per_ring
    lors = np.zeros((int(total_num_crystal*(total_num_crystal-1)/2),2))
    num = 0
    for i in range(total_num_crystal):
        for j in range(i):
            lors[num,0] = i
            lors[num,1] = j
            num = num+1
    return lors

def get_crystal_xy(grid_block,num_block,crystal_id,block_id,r_inner,size_block):
    size_crystal = size_block/grid_block
    angle_block = math.pi*2/num_block*block_id
    center = np.hstack((((r_inner+10)*np.cos(angle_block)).reshape(block_id.size,1),((r_inner+10)*np.sin(angle_block)).reshape(block_id.size,1)))
    vector_tang = np.hstack((np.cos(angle_block+math.pi/2*np.ones_like(angle_block)).reshape(block_id.size,1),np.sin(angle_block+math.pi/2*np.ones_like(angle_block)).reshape(block_id.size,1)))
    dist_from_center = size_crystal[1]*(crystal_id-(grid_block[1]-1)/2*np.ones_like(crystal_id))
    return center+np.hstack((dist_from_center.reshape(dist_from_center.size,1),dist_from_center.reshape(dist_from_center.size,1)))*vector_tang

def get_crystal_z(ring_id,grid_block,size_block,nb_rings):
    size_crystal = size_block/grid_block
    dist_from_center = (ring_id-(grid_block[2]*nb_rings-1)/2*np.ones_like(ring_id))*size_crystal[2]
    return dist_from_center

def get_center(scanner,crystal_id_whole_scanner):
    ring_id = np.floor(crystal_id_whole_scanner/scanner.nb_detectors_per_ring)
    crystal_per_ring_id = crystal_id_whole_scanner - ring_id*scanner.nb_detectors_per_ring
    block_id = crystal_per_ring_id//scanner.blocks[0].grid[1]
    crystal_id = crystal_per_ring_id%scanner.blocks[0].grid[1]
    center_xy = get_crystal_xy(np.array(scanner.blocks[0].grid),scanner.nb_blocks_per_ring,crystal_id,block_id,scanner.inner_radius,np.array(scanner.blocks[0].size))
    center_z = get_crystal_z(ring_id,np.array(scanner.blocks[0].grid),np.array(scanner.blocks[0].size),scanner.nb_rings)
    return np.hstack((center_xy,center_z.reshape(center_z.size,1)))

def lm2sino(filename,scanner):
    load_file = load_h5(filename)
    listmode_pos = np.hstack((load_file['fst'],load_file['snd']))
    listmode_id = detectorid(scanner,listmode_pos) 
    listmode_data = np.hstack((listmode_id,load_file['weight'].reshape(-1,1)))
    return cal_sinogram(listmode_data,scanner)

def sino2lm(scanner,sinogram,lors):
    index = np.where(sinogram>0)[0]
    all_position =  np.zeros((len(index),6))
    all_position[:,:3] = get_center(scanner,lors[index,0])
    all_position[:,3:6] = get_center(scanner,lors[index,1])
    return np.hstack((all_position,sinogram[index].reshape(-1,1)))