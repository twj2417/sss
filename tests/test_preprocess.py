from sss.preprocess import (pre_all_scatter_position,pre_sumup_of_emission,pre_atten,get_coordinate,
              get_image,get_voxel_volume,get_all_crystal_position,get_lors)
import srfnef as nef
from sss.scatter import get_scanner
import numpy as np
import pytest


def get_emission_image():
    data = np.zeros((100,100,100))
    center = np.array([0,0,0])
    size = np.array([100,150,200])
    return nef.Image(data,center,size)

def make_scanner():
    config = {"scanner": {
        "ring": {
            "inner_radius": 424.5,
            "outer_radius": 444.5,
            "axial_length": 220,
            "nb_rings": 1,
            "nb_blocks_per_ring": 4,
            "gap": 0.0
        },
        "block": {
            "grid": [1,1,1],
            "size": [20.0,53.3,53.3]
        }
    }
    }
    return get_scanner(config['scanner'])


def test_pre_all_scatter_position():
    emission_image = get_emission_image()
    scatter_position = np.zeros([10*17*20,3])
    step = np.array([10,6,5])
    for i in range(10):
        for j in range(17):
            for k in range(20):
                scatter_position[i*17*20+j*20+k,:] = get_coordinate(emission_image,step,i,j,k)
    assert (pre_all_scatter_position(emission_image)==scatter_position).all()

def test_pre_sumup_of_emission():
    pass

def test_pre_atten():
    pass

def test_get_lors():
    a = np.array([[1,2,3],[4,5,6]])
    b = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
    result = np.array([[1,2,3,0.1,0.2,0.3],[1,2,3,0.4,0.5,0.6],[4,5,6,0.1,0.2,0.3],[4,5,6,0.4,0.5,0.6]],dtype=np.float32)
    assert (get_lors(a,b).data == result).all()

def test_get_coordinate():
    emission_image = get_emission_image()
    step = np.array([5,5,5])
    position = np.array([-34.5,-36.75,-39])
    assert (get_coordinate(emission_image,step,3,5,6)==position).all()

def test_get_image():
    pass

def test_get_voxel_volume():
    emission_image = get_emission_image()
    assert get_voxel_volume(emission_image) == 3

def test_get_all_crystal_position():
    scanner = make_scanner()
    crystal_position = np.array([[434.5,0,0],[0,434.5,0],[-434.5,0,0],[0,-434.5,0]],dtype=np.float32)
    print(get_all_crystal_position(scanner))
    assert (get_all_crystal_position(scanner)==crystal_position).all()
