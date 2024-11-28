import os


def pro_root_path():
    """project root path"""
    return os.path.abspath(os.curdir)


def rs_models_path():
    return os.path.join(pro_root_path(), 'hiera_diffusion_policy/env/nonprehensile/rsuite/models')


def rs_assets_path():
    return os.path.join(rs_models_path(), 'assets')



if __name__ == '__main__':
    print(rs_models_path())