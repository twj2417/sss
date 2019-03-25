import numpy as np
import math
import srfnef as nef
from .listmode2sinogram.listmode2sinogram import get_center


def pre_all_scatter_position(emission_image):
    size_voxel = emission_image.size/emission_image.shape
    step = np.floor(50/size_voxel)
    time = np.floor(emission_image.shape/step)
    position = np.zeros((int(time[0]*time[1]*time[2]),3),dtype=np.float32)
    n = 0
    for x in range(int(time[0])):
        for y in range(int(time[1])):
            for z in range(int(time[2])):
                position[n,:] = get_coordinate(emission_image,step,x,y,z)
                n = n+1
    return position

def pre_sumup_of_emission(emission_image,p1,p2,config):
    projector = nef.projector_picker('siddon')('gpu')
    lors = get_lors(p1,p2)
    value = projector(emission_image,lors).data
    len_lors = np.power(lors.data[:,3]-lors.data[:,0],2)+np.power(lors.data[:,4]-lors.data[:,1],2)+np.power(lors.data[:,5]-lors.data[:,2],2)
    pixel_size = emission_image.size[0]/emission_image.data.shape[0]
    return value*len_lors*2

def pre_atten(transmission_image,p1,p2,config):
    projector = nef.projector_picker('siddon')('gpu')
    lors = get_lors(p1,p2)
    value = projector(transmission_image,lors).data
    len_lors = np.power(lors.data[:,3]-lors.data[:,0],2)+np.power(lors.data[:,4]-lors.data[:,1],2)+np.power(lors.data[:,5]-lors.data[:,2],2)
    pixel_size = transmission_image.size[0]/transmission_image.data.shape[0]
    weight = np.exp(-value*len_lors*2)
    return weight

def pre_lors_atten(transmission_image,p1,p2,config):
    projector = nef.projector_picker('siddon')('gpu')
    lors = np.hstack((np.hstack((p1,p2)),np.ones((p1.shape[0],1),dtype=np.float32)))
    value = projector(transmission_image,nef.LORs(lors)).data
    len_lors = np.power(lors[:,3]-lors[:,0],2)+np.power(lors[:,4]-lors[:,1],2)+np.power(lors[:,5]-lors[:,2],2)
    pixel_size = transmission_image.size[0]/transmission_image.data.shape[0]
    weight = np.exp(-value*len_lors*2)
    return weight

def get_lors(P1,P2):
    lors = np.ones((P1.shape[0]*P2.shape[0],7),dtype=np.float32)
    lors[:,0] = np.tile(P1[:,0].reshape(-1,1),P2.shape[0]).flatten()
    lors[:,1] = np.tile(P1[:,1].reshape(-1,1),P2.shape[0]).flatten()
    lors[:,2] = np.tile(P1[:,2].reshape(-1,1),P2.shape[0]).flatten()
    lors[:,3] = np.tile(P2[:,0],P1.shape[0])
    lors[:,4] = np.tile(P2[:,1],P1.shape[0])
    lors[:,5] = np.tile(P2[:,2],P1.shape[0])
    return nef.LORs(lors)

def get_coordinate(emission_image,step,x,y,z):
    size_pixel = emission_image.size/emission_image.shape
    return np.array([step[0]*(x+1/2)*size_pixel[0],step[1]*(y+1/2)*size_pixel[1],step[2]*(z+1/2)*size_pixel[2]])-emission_image.size/2

def get_image(config):
    image_data = np.load(config['path']).astype(np.float32)
    center = config['center']
    size = config['size']
    return nef.Image(image_data,center,size)

def get_voxel_volume(image):
    size_voxel = image.size/image.shape
    return size_voxel[0]*size_voxel[1]*size_voxel[2]

def get_all_crystal_position(scanner):
    crystal_id = np.arange(scanner.nb_detectors)
    return get_center(scanner,crystal_id).astype(np.float32)

