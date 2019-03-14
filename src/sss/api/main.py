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
    start0 = time.time()
    start = time.time()
    scanner = get_scanner(config['scanner'])
    lors = get_all_lors_id(scanner.nb_rings*scanner.blocks.shape[2]*scanner.nb_detectors_per_ring)
    sinogram = lm2sino(config['listmode']['path'],scanner)
    end = time.time()
    print(end-start)
    start = time.time()
    index = np.where(sinogram>0)[0].astype(np.int32)
    fraction,scale,atten = get_scatter_fraction(config,index,lors,scanner)
    end = time.time()
    print(end-start)
    start = time.time()
    corrected_sinogram = np.zeros_like(sinogram)
    corrected_sinogram[index] = (sinogram[index]-fraction/np.sum(fraction)*np.sum(sinogram[index])*0.44)*scale/(atten+sys.float_info.min)
    corrected_data = sino2lm(scanner,corrected_sinogram,lors)
    end = time.time()
    print(end-start)
    start = time.time()
    result = {'fst':corrected_data[:,0:3],'snd':corrected_data[:,3:6],'weight':corrected_data[:,6]}
    save_h5(config['output']['path'],result)
    end = time.time()
    end0 = time.time()
    print(end0-start0)