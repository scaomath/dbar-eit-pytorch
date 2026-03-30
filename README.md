# D-bar Method for Electrical Impedance Tomography in PyTorch 

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This repository has an implementation of D-bar method (both solve and approximation) natively in PyTorch, compatible with `torch.autograd`.



## Instructions
You need to download the `KIT4_Dbar_recon.zip` data file from the official MATLAB implementation of D-bar method at https://fips.fi/blog/the-d-bar-method-for-electrical-impedance-tomography-experimental-data/ and https://arxiv.org/pdf/1704.01178, then the data should be extracted to `data` and `KIT4_measdata` folders.











## References
Our implementation is based on the official implementation based on KIT4 dataset:  https://fips.fi/blog/the-d-bar-method-for-electrical-impedance-tomography-experimental-data/ and their [wiki page](https://wiki.helsinki.fi/xwiki/bin/view/mathstatHenkilokunta/Henkil%C3%B6t/Siltanen%2C%20Samuli/Inverse%20Problems%20Book%20Page/EIT%20with%20the%20D-bar%20method%3A%20discontinuous%20heart-and-lungs%20phantom/).
The initial porting draft of the MATLAB code based on KIT4 on circular domain is done by GPT 5.4 using VSCode Copilot. The `numpy`-based iterative method in https://github.com/eitcom/pyEIT is also used as a reference code.

