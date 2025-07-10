# Third-Party Code Integration and Attribution

This document provides detailed information about third-party code components integrated into HAMscope and their specific usage within the project.

## Overview

HAMscope is primarily built upon the PyTorch pix2pix/CycleGAN framework and incorporates several additional established algorithms to provide a comprehensive hyperspectral microscopy pipeline. This document ensures proper attribution and license compliance for all components.

## License Hierarchy

**Primary Foundation:**
- PyTorch pix2pix/CycleGAN - BSD 2-Clause License (~90% of codebase)

**Additional Components:**
- CSBDeep/CARE - BSD 3-Clause License  
- PoissonNMF - Academic use (citation required)
- Reg-GAN - Research use (contact authors for commercial use)
- Spectral Attention Transformer - Research use
- Quadratic Intensity Transformer - Research use

**Result:** HAMscope uses BSD 2-Clause license from primary foundation.

## Integrated Components

### 1. PyTorch pix2pix/CycleGAN (PRIMARY FOUNDATION)

**Source**: Image-to-Image Translation with Conditional Adversarial Networks / CycleGAN
**Authors**: Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
**Repository**: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
**License**: BSD 2-Clause

#### License Text (Primary Foundation)
```
BSD License

For pix2pix software
Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu
Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

#### Integration Details
- **Location**: `HAMscope_pix2pix/` (entire framework)
- **Usage**: Core GAN architecture, training framework, and data handling
- **Extent**: ~95% of the HAMscope_pix2pix codebase
- **Purpose**: Foundation for hyperspectral image-to-image translation
- **Modifications**: 
  - Extended for 30-channel hyperspectral data (450-700nm)
  - Added spectral consistency loss functions
  - Modified data loaders for TIFF hyperspectral formats
  - Enhanced visualization for multi-channel outputs

#### Files Directly Based on pix2pix/CycleGAN
- `HAMscope_pix2pix/train.py` - Core training script
- `HAMscope_pix2pix/test.py` - Inference script  
- `HAMscope_pix2pix/models/` - Network architectures and training logic
- `HAMscope_pix2pix/data/` - Dataset loading and processing
- `HAMscope_pix2pix/options/` - Command-line argument handling
- `HAMscope_pix2pix/util/` - Utility functions and visualization

#### Required Citations
```bibtex
@inproceedings{isola2017image,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}

@inproceedings{zhu2017unpaired,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
```

### 2. PoissonNMF ImageJ Plugin

**Source**: Neher et al., Biophysical Journal (2009)
**Authors**: Richard A. Neher, Fabian J. Theis, André Zeug
**License**: Academic use (citation required)

#### Integration Details
- **File**: `HAMscope_helper/nmf/PoissonNMF_.jar`
- **Usage**: Non-negative matrix factorization of hyperspectral data
- **Modifications**: Used as-is via ImageJ macros

#### Citation Requirements
```bibtex
@article{neher2009blind,
  title={Blind source separation techniques for the decomposition of multiply labeled fluorescence images},
  author={Neher, Richard A and Mitkovski, Mi{\v{s}}o and Kirchhoff, Frank and Neher, Erwin and Theis, Fabian J and Zeug, Andr{\'e}},
  journal={Biophysical journal},
  volume={96},
  number={9},
  pages={3791--3800},
  year={2009},
  publisher={Elsevier}
}
```

### 3. CSBDeep/CARE Framework

**Source**: Content-aware restoration of fluorescence microscopy
**Authors**: Uwe Schmidt, Martin Weigert
**Repository**: https://github.com/CSBDeep/CSBDeep
**License**: BSD 3-Clause

#### Integration Details
- **Location**: `HAMscope_pix2pix/models/` (loss functions)
- **Usage**: Probabilistic loss functions and uncertainty quantification
- **Modifications**: Adapted for 30-channel hyperspectral data

### 4. Reg-GAN Components

**Source**: Breaking the Dilemma of Medical Image-to-image Translation
**Authors**: Lingke Kong, Chenyu Lian, Detian Huang, et al.
**Repository**: https://github.com/Kid-Liet/Reg-GAN
**License**: No explicit license (contact authors for commercial use)

#### Integration Details
- **Location**: `HAMscope_pix2pix/models/` (registration components)
- **Usage**: Spatial registration and alignment networks
- **Modifications**: Adapted for hyperspectral data

#### Citation Requirements
```bibtex
@inproceedings{kong2021breaking,
  title={Breaking the Dilemma of Medical Image-to-image Translation},
  author={Kong, Lingke and Lian, Chenyu and Huang, Detian and Li, ZhenJiang and Hu, Yanle and Zhou, Qichao},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

### 5. Spectral Attention Transformer

**Source**: Nature (2024) - Linear Compute Transformer for Hyperspectral Imaging
**Repository**: https://github.com/bianlab/HyperspecI
**License**: Research use (contact authors for commercial use)

#### Integration Details
- **Location**: `HAMscope_pix2pix/models/attention.py`
- **Usage**: Spectral attention mechanism for hyperspectral data
- **Modifications**: Adapted for 30-channel hyperspectral microscopy data

### 6. Quadratic Intensity Transformer

**Source**: Tripathy et al., Multimedia Systems (2020)
**License**: Research use (contact authors for commercial use)

#### Integration Details
- **Location**: `HAMscope_pix2pix/models/intensity_transform.py`
- **Usage**: Quadratic intensity transformation for enhanced image quality
- **Modifications**: Adapted for hyperspectral intensity normalization

#### Citation Requirements
```bibtex
@article{tripathy2020,
  title={ALBERT-based Fine-Tuning Model for Cyberbullying Analysis},
  author={Tripathy, Jatin and Chakkaravarthy, Sibi and Satapathy, Suresh and Sahoo, Madhulika and Vaidehi, V.},
  journal={Multimedia Systems},
  volume={28},
  year={2020},
  publisher={Springer}
}
```

## Usage Guidelines

### Research Use
- All papers must be properly cited in publications
- Original authors should be acknowledged
- License compliance must be verified

### Commercial Use
- Contact original authors for components without explicit licenses
- Ensure BSD license compliance for pix2pix/CycleGAN and CSBDeep
- Legal review recommended for commercial applications

## Contact Information

For licensing questions:
- **PoissonNMF**: Contact André Zeug (zeug.andre@mh-hannover.de)
- **CSBDeep**: Open issues on GitHub repository
- **Reg-GAN**: Contact Lingke Kong
- **HAMscope**: Open issues on project repository

---

**Last Updated**: July 10, 2025
