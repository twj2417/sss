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
    scanner = get_scanner(config['scanner'])
    lors = get_all_lors_id(scanner.nb_rings*scanner.nb_detectors_per_ring)
    sinogram = lm2sino(config['listmode']['path'],scanner)
    index = np.where(sinogram>0)[0].astype(np.int32)
    atten = get_scatter_fraction(config,index,lors,scanner)
    corrected_sinogram = np.zeros_like(sinogram)
    corrected_sinogram[index] = sinogram[index]/(atten+sys.float_info.min)
    corrected_data = sino2lm(scanner,corrected_sinogram,lors)
    end0 = time.time()
    print(end0-start0)
    result = {'fst':corrected_data[:,0:3],'snd':corrected_data[:,3:6],'weight':corrected_data[:,6]}
    save_h5(config['output']['path'],result)
    