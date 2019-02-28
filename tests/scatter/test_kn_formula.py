from sss.scatter.kn_formula import fkn,get_scatter_cos_theta
import pytest


def test_get_scatter_cos_theta():
    A = [1,0,0]
    S = [0,0,0]
    B = [0,1,0]
    assert abs(get_scatter_cos_theta(A,S,B)) < 10**(-15)

def test_fkn():
    pass
