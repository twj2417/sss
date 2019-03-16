import numpy as np
import math
from numba import jit,cuda
import h5py
from typing import Dict

@cuda.jit(device=True)
def norm(x,y):
    return (x**2+y**2)**0.5

@cuda.jit(device=True)
def get_block_theta(event):
    before_theta = event[0]/norm(event[0],event[1])
    return math.acos(before_theta)/math.pi*180

@cuda.jit(device=True)
def block_id(event,nb_detectors_per_ring,block_grid_y,num_block):
    theta_event = get_block_theta(event)
    if event[1]<0:
        theta_event = 360-theta_event
    fixed_theta = (theta_event+180/nb_detectors_per_ring*block_grid_y)%360
    return math.floor(fixed_theta/360*num_block)

@cuda.jit(device=True)
def crystal_id(nb_detectors_per_ring,nb_blocks_per_ring,inner_radius,outer_radius,grid_block,size_block,event):
    id_block = block_id(event,nb_detectors_per_ring,grid_block[1],nb_blocks_per_ring)
    block_center_x = (inner_radius+outer_radius)/2*math.cos(2*math.pi*id_block/nb_blocks_per_ring)
    block_center_y = (inner_radius+outer_radius)/2*math.sin(2*math.pi*id_block/nb_blocks_per_ring)
    theta_block = 2*math.pi*id_block/nb_blocks_per_ring
    dist = (block_center_x-event[0])*math.cos(theta_block)+(block_center_y-event[1])*math.sin(theta_block)
    pro_point_x = dist*math.cos(theta_block)+event[0]
    pro_point_y = dist*math.sin(theta_block)+event[1]
    edge_block_x = block_center_x - math.cos(theta_block+math.pi/2)*size_block[1]/2
    edge_block_y = block_center_y - math.sin(theta_block+math.pi/2)*size_block[1]/2
    id_crystal = int(norm(pro_point_x-edge_block_x,pro_point_y-edge_block_y)/size_block[1]*grid_block[1])
    return id_block*grid_block[1]+id_crystal
    
@cuda.jit(device=True)
def ring_id(nb_rings,axial_length,grid_block,size_block,event_z):
    return int((event_z+size_block[2]*nb_rings/2)/axial_length*nb_rings*grid_block[1])

@cuda.jit(device=True)
def position2detectorid(nb_detectors_per_ring,nb_blocks_per_ring,nb_rings,inner_radius,outer_radius,axial_length,grid_block,
                        size_block,event):
    ring_id1 = ring_id(nb_rings,axial_length,grid_block,size_block,event[2])
    crystal_id1 = crystal_id(nb_detectors_per_ring,nb_blocks_per_ring,inner_radius,outer_radius,grid_block,size_block,event[:2])
    ring_id2 = ring_id(nb_rings,axial_length,grid_block,size_block,event[5])
    crystal_id2 = crystal_id(nb_detectors_per_ring,nb_blocks_per_ring,inner_radius,outer_radius,grid_block,size_block,event[3:5])
    return (ring_id1*nb_detectors_per_ring+crystal_id1),(ring_id2*nb_detectors_per_ring+crystal_id2)

@cuda.jit(device=True)
def change_id(id1,id2):
    if id1<id2:
        return id2,id1
    else:
        return id1,id2

@cuda.jit
def detectorid(nb_detectors_per_ring,nb_blocks_per_ring,nb_rings,inner_radius,outer_radius,axial_length,grid_block,size_block,events,final_id):
    c,j = cuda.grid(2)
    i = 512*16*c+j
    if i<events.shape[0]:
        id1,id2 = position2detectorid(nb_detectors_per_ring,nb_blocks_per_ring,nb_rings,inner_radius,outer_radius,axial_length,
                                      grid_block,size_block,events[i,:])
        fixed_id1,fixed_id2 = change_id(id1,id2)
        final_id[i,0] = fixed_id1
        final_id[i,1] = fixed_id2

@jit(nopython=True)
def cal_sinogram(list_mode_data,nb_rings,grid_block_z,nb_detectors_per_ring):
    total_num_crystal = nb_rings*grid_block_z*nb_detectors_per_ring
    weight = np.zeros((int(total_num_crystal*(total_num_crystal-1)/2),1))
    for i in range(int(list_mode_data.shape[0])):
        pos = int((list_mode_data[i,0]-1)*list_mode_data[i,0]/2+list_mode_data[i,1])
        weight[pos] +=  list_mode_data[i,2]
    return weight

@jit(nopython=True)
def get_all_lors_id(total_num_crystal):
    lors = np.zeros((int(total_num_crystal*(total_num_crystal-1)/2),2),dtype=np.int32)
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
    center = np.hstack((((r_inner)*np.cos(angle_block)).reshape(block_id.size,1),((r_inner)*np.sin(angle_block)).reshape(block_id.size,1)))
    vector_tang = np.hstack((np.cos(angle_block+math.pi/2*np.ones_like(angle_block)).reshape(block_id.size,1),
                             np.sin(angle_block+math.pi/2*np.ones_like(angle_block)).reshape(block_id.size,1)))
    dist_from_center = size_crystal[1]*(crystal_id-(grid_block[1]-1)/2*np.ones_like(crystal_id))
    return center+(np.hstack((dist_from_center.reshape(dist_from_center.size,1),dist_from_center.reshape(dist_from_center.size,1)))
                   *vector_tang)

def get_crystal_z(ring_id,grid_block,size_block,nb_rings):
    size_crystal = size_block/grid_block
    dist_from_center = (ring_id-(grid_block[2]*nb_rings-1)/2*np.ones_like(ring_id))*size_crystal[2]
    return dist_from_center

def get_center(scanner,crystal_id_whole_scanner):
    ring_id = np.floor(crystal_id_whole_scanner/scanner.nb_blocks_per_ring/scanner.blocks.shape[1])
    crystal_per_ring_id = crystal_id_whole_scanner - ring_id*scanner.nb_blocks_per_ring*scanner.blocks.shape[1]
    block_id = crystal_per_ring_id//scanner.blocks.shape[1]
    crystal_id = crystal_per_ring_id%scanner.blocks.shape[1]
    center_xy = get_crystal_xy(np.array(scanner.blocks.shape),scanner.nb_blocks_per_ring,crystal_id,block_id,scanner.inner_radius,
                               np.array(scanner.blocks.size))
    center_z = get_crystal_z(ring_id,np.array(scanner.blocks.shape),np.array(scanner.blocks.size),scanner.nb_rings)
    return np.hstack((center_xy,center_z.reshape(center_z.size,1)))

@jit
def lm2sino(filename,scanner):
    load_file = load_h5(filename)
    listmode_pos = np.hstack((load_file['fst'],load_file['snd']))
    listmode_id = np.zeros((listmode_pos.shape[0],2),dtype=np.float32)
    detectorid[(512,512),(16,16)](scanner.nb_blocks_per_ring*scanner.blocks.shape[1],scanner.nb_blocks_per_ring,scanner.nb_rings,scanner.inner_radius,
            scanner.outer_radius,scanner.axial_length,np.array(scanner.blocks.shape,dtype=np.int32),
            np.array(scanner.blocks.size,dtype=np.float32),listmode_pos,listmode_id) 
    listmode_data = np.hstack((listmode_id,load_file['weight'].reshape(-1,1)))
    return cal_sinogram(listmode_data,scanner.nb_rings,scanner.blocks.shape[2],scanner.nb_blocks_per_ring*scanner.blocks.shape[1])

def sino2lm(scanner,sinogram,lors):
    index = np.where(sinogram>0)[0]
    all_position =  np.zeros((len(index),6))
    all_position[:,:3] = get_center(scanner,lors[index,0])
    all_position[:,3:6] = get_center(scanner,lors[index,1])
    return np.hstack((all_position,sinogram[index].reshape(-1,1)))


__all__ = []

DEFAULT_GROUP_NAME = 'listmode_data'

DEFAULT_COLUMNS = ['fst', 'snd', 'weight', 'tof']


def load_h5(path, group_name=DEFAULT_GROUP_NAME)-> Dict[str, np.ndarray]:
    with h5py.File(path, 'r') as fin:
        dataset = fin[group_name]
        result = {}
        for k in DEFAULT_COLUMNS:
            if k in dataset:
                result[k] = np.array(dataset[k])
        return result

def save_h5(path, dct, group_name=DEFAULT_GROUP_NAME):
    with h5py.File(path, 'w') as fout:
        group = fout.create_group(group_name)
        for k, v in dct.items():
            group.create_dataset(k, data=v, compression="gzip")
