# render templates
import os

template_list = ['beinsum.template.f90']
source_list = [tmplt[:-12] + 'f90' for tmplt in template_list]
extension_list = [source[:-4] for source in source_list]

libdir = os.path.dirname(__file__)

def render_f90s(templates=None):
    from frender import render_f90
    if templates is None:
        templates = template_list
    else:
        templates = templates
    for template in templates:
        source = template[:-12] + 'f90'
        pytime = os.path.getmtime(os.path.join(libdir, 'templates', template))
        source_file = os.path.join(libdir, source)
        if not os.path.isfile(source_file) or \
                os.path.getmtime(source_file) < pytime:
            render_f90(libdir, os.path.join('templates', template), {
                'dtype_list': ['complex*16', 'complex*8', 'real*8', 'real*4']
            }, out_file=os.path.join(libdir, source))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError, numpy_info
    config = Configuration('lib', parent_package, top_path)

    # get lapack options
    lapack_opt = get_info('lapack_opt')

    if not lapack_opt:
        raise NotFoundError('no lapack/blas resources found')

    atlas_version = ([v[3:-3] for k, v in lapack_opt.get('define_macros', [])
                      if k == 'ATLAS_INFO'] + [None])[0]
    if atlas_version:
        print(('ATLAS version: %s' % atlas_version))

    # include_dirs=[os.curdir,'$MKLROOT/include']
    # library_dirs=['$MKLROOT/lib/intel64']
    # libraries=['mkl_intel_lp64','mkl_sequential','mkl_core', 'm', 'pthread']

    # render f90 files if templates changed
    render_f90s()

    for extension, source in zip(extension_list, source_list):
        # config.add_extension(
        # extension, [os.path.join(libdir, source)], libraries=libraries,
        #        library_dirs=library_dirs, include_dirs=include_dirs)
        config.add_extension(extension, [os.path.join(
            libdir, source)], extra_info=lapack_opt)
    return config
