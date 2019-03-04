import click
import json
from ..api.main import scatter_correction
from dxl.core.debug import enter_debug
from srf.external.stir.function import get_scanner

enter_debug()

@click.command()
@click.option('--config','-c',type=click.Path(exists=True))
def sss(config):
    with open(config, 'r') as fin:
        task_config = json.load(fin)
    scatter_correction(task_config)