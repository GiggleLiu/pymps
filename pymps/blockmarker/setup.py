#from numpy.distutils.core import setup, Extension
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('blockmarker', parent_package, top_path)
    config.add_extension('fblock', ['fblock.f90'])
    return config
