from ..io.load_data import get_data
from ..listmode2sinogram import listmode2sinogram
from ..scatter import get_scatter_fraction

def scatter_correction(config):
    data = get_data()
    sinogram = listmode2sinogram(data)
    fraction = get_scatter_fraction()
    corrected_sinogram = sinogram*fraction
    return corrected_sinogram