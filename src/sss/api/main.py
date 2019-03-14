from ..io.load_data import get_data
from ..listmode2sinogram.listmode2sinogram import lm2sino,sino2lm,get_all_lors_id,save_h5
from ..scatter import get_scatter_fraction,get_scanner
import numpy as np
import time
import sys
np.seterr(divide='ignore', invalid='ignore')
# from srf.external.stir.function import get_scanner
# from srf.io.listmode import load_h5,save_h5

def scatter_correction(config):
    scanner = get_scanner(config['scanner'])
    lors = get_all_lors_id(scanner.nb_rings*scanner.blocks.shape[2]*scanner.nb_detectors_per_ring)
    sinogram = lm2sino(config['listmode']['path'],scanner)
    fraction,scale,atten = get_scatter_fraction(config,sinogram,lors,scanner)
    corrected_sinogram = (sinogram-fraction)*scale/(atten+sys.float_info.min)
    corrected_data = sino2lm(scanner,corrected_sinogram,lors)
    result = {'fst':corrected_data[:,0:3],'snd':corrected_data[:,3:6],'weight':corrected_data[:,6]}
    save_h5(config['output']['path'],result)