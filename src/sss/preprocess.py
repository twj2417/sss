from srf.data import Image
import numpy as np
import math
from dxl.learn.core import Session, SubgraphMakerTable
from srf.graph.reconstruction.attenuation import Attenuation
from srf.model import ProjectionOrdinary
from srf.physics import CompleteLoRsModel
from srf.data.listmode import ListModeData
from .listmode2sinogram.listmode2sinogram import get_center
import sys

def pre_all_scatter_position(emission_image):
    size_voxel = emission_image.size/emission_image.grid
    step = np.floor(50/size_voxel)
    time = np.ceil(emission_image.grid/step)
    position = np.zeros((int(time[0]*time[1]*time[2]),3))
    n = 0
    for x in range(int(time[0])):
        for y in range(int(time[1])):
            for z in range(int(time[2])):
                position[n,:] = get_coordinate(emission_image,step,x,y,z)
                n = n+1
    return position

def pre_sumup_of_emission(emission_image,p1,p2,config):
    model = CompleteLoRsModel('model',**config['algorithm']['projection_model']['siddon'])
    projection = ProjectionOrdinary(model)
    lors = get_lors(p1,p2)
    projection_data = ListModeData(lors, np.ones([lors.shape[0]], np.float32))
    g = Attenuation('attenuation',projection,projection_data,emission_image.data,emission_image.center,emission_image.size)
    g.make()
    with Session() as sess:
        value = g.run(sess)
    sess.reset()
    return value

def pre_atten(emission_image,u_map,p1,p2,config):
    model = CompleteLoRsModel('model',**config['algorithm']['projection_model']['siddon'])
    projection = ProjectionOrdinary(model)
    lors = get_lors(p1,p2)
    projection_data = ListModeData(lors, np.ones([lors.shape[0]], np.float32))
    g = Attenuation('attenuation',projection,projection_data,u_map.data,u_map.center,u_map.size)
    g.make()
    with Session() as sess:
        value = g.run(sess)
    sess.reset()
    len_lors = np.power(lors[:,3]-lors[:,0],2)+np.power(lors[:,4]-lors[:,1],2)+np.power(lors[:,5]-lors[:,2],2)
    weight = 1.0/(np.exp(-value*len_lors)+sys.float_info.min)
    return weight

def get_lors(P1,P2):
    lors = np.ones((P1.shape[0]*P2.shape[0],7))
    lors[:,0] = np.tile(P1[:,0].reshape(-1,1),P2.shape[0]).flatten()
    lors[:,1] = np.tile(P1[:,1].reshape(-1,1),P2.shape[0]).flatten()
    lors[:,2] = np.tile(P1[:,2].reshape(-1,1),P2.shape[0]).flatten()
    lors[:,3] = np.tile(P2[:,0],P1.shape[0])
    lors[:,4] = np.tile(P2[:,1],P1.shape[0])
    lors[:,5] = np.tile(P2[:,2],P1.shape[0])
    return lors

def get_coordinate(emission_image,step,x,y,z):
    size_pixel = emission_image.size/emission_image.grid
    return np.array([(step[0]*x+1/2)*size_pixel[0],(step[1]*y+1/2)*size_pixel[1],(step[2]*z+1/2)*size_pixel[2]])-emission_image.size/2

def get_image(config):
    image_data = np.load(config['path'])
    center = np.array(config['center'])
    size = np.array(config['size'])
    return Image(image_data,center,size)

def get_voxel_volume(image):
    size_voxel = image.size/image.grid
    return size_voxel[0]*size_voxel[1]*size_voxel[2]

def get_all_crystal_position(scanner):
    crystal_id = np.arange(scanner.nb_detectors)
    return get_center(scanner,crystal_id)

