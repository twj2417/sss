from sss.scatter.scatter_fraction import scatter_fraction,loop_all_s,eff_with_scatter,eff_without_scatter,project_area,get_block_theta,distance_a2b
from srf.external.stir.function import get_scanner
import numpy as np
import pytest

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

def test_scatter_fraction():
    pass

def test_loop_all_s():
    pass

def test_eff_with_scatter():
    assert eff_with_scatter(511,511,1,50) == 0

def test_eff_without_scatter():
    assert eff_without_scatter(511,511,50) == 0

def test_project_area():
    scanner = make_scanner()
    pa = np.array([0,0,0])
    pb = np.array([434.5,0,0])
    assert project_area(scanner,pa,pb) == 2840.89

def test_get_block_theta():
    scanner = make_scanner()
    p = np.array([430.3,22.2,10.1])
    theta = get_block_theta(scanner,p)
    assert theta == 0


def test_distance_a2b():
    assert abs(distance_a2b(1,3,5,9,7.7,0.1)-10.49285)<10**-5
