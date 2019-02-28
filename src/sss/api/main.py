from ..io.load_data import get_data
from ..listmode2sinogram.listmode2sinogram import lm2sino,sino2lm,get_all_lors_id
from ..scatter import get_scatter_fraction
from srf.external.stir.function import get_scanner
from srf.io.listmode import load_h5,save_h5

def scatter_correction(config):
    scanner = get_scanner(config['scanner'])
    lors = get_all_lors_id(scanner.nb_rings*scanner.blocks[0].grid[2]*scanner.nb_detectors_per_ring)
    sinogram = lm2sino(config['listmode']['path'],scanner)
    fraction = get_scatter_fraction(config)
    corrected_sinogram = sinogram*(1-fraction)
    corrected_data = sino2lm(scanner,corrected_sinogram,lors)
    result = {'fst':corrected_data[:,0:3],'snd':corrected_data[:,3:6],'weight':corrected_data[:,6]}
    save_h5(config['output']['path'],result)