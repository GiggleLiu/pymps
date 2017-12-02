from jinja2 import Template
from jinja2 import Environment, FileSystemLoader, select_autoescape


def render_f90(template_folder, template_file, var_dict, out_file):
    env = Environment(
        loader=FileSystemLoader(template_folder),
    )

    res = env.get_template(template_file).render(var_dict)
    with open(out_file, 'w') as of:
        of.write('!This is an f90 file automatically generated.\n' + res)


if __name__ == '__main__':
    render_f90('templates', 'spconv_cc.template.f90', {'version_list': [
               ''], 'dtype_list': ['complex*16', 'real*8', 'real*4']},
               out_file='spconv_cc.f90')
    render_f90('templates', 'spsp.template.f90', {'dtype_list': [
               'complex*16', 'real*8', 'real*4'],
               'version_list': ['', '_conv']}, out_file='spsp.f90')
    render_f90('templates', 'spconv.template.f90', {'version_list': [
               'general', 'contiguous'],
               'dtype_list': ['complex*16', 'real*8', 'real*4']},
               out_file='spconv.f90')
    render_f90('templates', 'linear.template.f90', {'version_list': [
               ''], 'dtype_list': ['complex*16', 'real*8', 'real*4']},
               out_file='linear.f90')
    render_f90('templates', 'pooling.template.f90', {'version_list': [
               ''], 'dtype_list': ['complex*16', 'real*8', 'real*4']},
               out_file='pooling.f90')
    render_f90('templates', 'relu.template.f90', {'version_list': [
               ''], 'dtype_list': ['complex*16', 'real*8', 'real*4']},
               out_file='relu.f90')
