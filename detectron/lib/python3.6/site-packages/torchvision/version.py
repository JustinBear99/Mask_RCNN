__version__ = '0.6.1'
git_version = '35d732ac53aebbed917993523d685b4cb09ef6ea'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
