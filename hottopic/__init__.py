
from . import util

from . import dataset
from . import rawdata
from . import augment
from . import sample
from . import preprocess
from . import model
from . import conv

try:
    import matplotlib
except Exception as e:
    print('Not going to import the visualization modules: {}'.format(e))
if 'matplotlib' in globals():
    from . import viz
    from .viz import windrose
    from .viz import render
    from .viz import async
    from .viz import gui
    from .viz import dayviewer
    print('Successfully imported the visualization modules')
