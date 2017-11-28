'''
Set up file for matrix product state.
'''


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('pymps', parent_package, top_path)
    config.add_subpackage('tensor')
    config.add_subpackage('ansatz')
    config.add_subpackage('blockmarker')
    config.add_subpackage('construct')
    config.add_subpackage('apps')
    config.add_subpackage('toolbox')
    return config
