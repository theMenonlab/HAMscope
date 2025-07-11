# HAMscope - Hyperspectral Microscopy Pipeline

A comprehensive toolkit for hyperspectral microscopy data processing, analysis, and AI-based enhancement using CycleGAN and pix2pix models.

## Overview

HAMscope is a complete pipeline for working with hyperspectral microscopy data, particularly designed for biological imaging applications. Built upon the PyTorch pix2pix/CycleGAN framework by Jun-Yan Zhu and Taesung Park, the project combines traditional image processing techniques with modern deep learning approaches to enhance hyperspectral microscopy data quality and extract meaningful biological information.

The system extends the original pix2pix architecture to handle 30-channel hyperspectral data (450-700nm) and incorporates specialized components for spectral analysis, uncertainty quantification, and spatial registration.

## Key Features

- **Hyperspectral Data Processing**: Complete pipeline for hyperspectral image preprocessing, normalization, and format conversion
- **AI-Enhanced Reconstruction**: Modified pix2pix models optimized for hyperspectral data
- **Spectral Analysis**: Non-negative matrix factorization (NMF) and spectral unmixing tools
- **Visualization Tools**: Comprehensive visualization utilities for hyperspectral data
- **Time-lapse Support**: Tools for creating movies from hyperspectral time series
- **3D Printing Support**: STL files for custom hardware components
- **Deconvolution**: Ring deconvolution microscopy (RDM) implementation

## Project Structure

```
HAMscope/
├── HAMscope_helper/          # Utility scripts and analysis tools
├── HAMscope_pix2pix/         # Modified pix2pix implementation for hyperspectral data
├── stl_files/                # 3D printable hardware components
└── README.md                 # This file
```

## Hardware Components

The `stl_files/` directory contains 3D printable components for the HAMscope hardware:

- **LED Brackets**: Mounting hardware for LED illumination systems
- **Mirror Spinners**: Motorized mirror rotation components
- **Filter Holders**: Liquid variable bandpass filter (LVBF) mounting
- **Plant Clamps**: Specialized clamps for plant imaging experiments

## Key Applications

- **Plant Biology**: Hyperspectral imaging of plant tissues for stress detection and physiological studies
- **Cell Biology**: High-resolution spectral imaging of cellular components
- **Material Science**: Spectral analysis of material properties
- **Biomedical Research**: Tissue characterization and disease detection

## Spectral Configuration

The system is optimized for:
- **Wavelength Range**: 450-700 nm
- **Channels**: 30 spectral channels
- **Spatial Resolution**: 512x512 pixels (configurable)
- **Bit Depth**: 16-bit per channel

## Documentation

Each subdirectory contains detailed documentation:
- [HAMscope_helper/README.md](HAMscope_helper/README.md) - Utility scripts and analysis tools
- [HAMscope_pix2pix/README.md](HAMscope_pix2pix/README.md) - AI model training and inference
- [stl_files/README.md](stl_files/README.md) - Hardware component specifications
- [THIRD_PARTY_ATTRIBUTION.md](THIRD_PARTY_ATTRIBUTION.md) - Detailed third-party code attribution and licensing

## Contributing

We welcome contributions! Please read our contributing guidelines and submit pull requests for any enhancements.

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

**Note**: This project is built upon the PyTorch pix2pix/CycleGAN implementation and inherits its BSD license terms.

## Citation

If you use HAMscope in your research, please cite:

```bibtex
@article{hamscope2025,
  title={HAMscope: a snapshot Hyperspectral Autofluorescence Miniscope for real-time molecular imaging in living plants},
  author={Alexander Ingold, et. al},
  journal={Submitted to Nature Methods},
  year={2025}
}
```

### Key References

**PoissonNMF Algorithm:**
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

**Original pix2pix:**
```bibtex
@inproceedings{isola2017image,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}
```


## Acknowledgments

- **Primary Foundation**: Built upon the PyTorch implementation of CycleGAN and pix2pix by Jun-Yan Zhu and Taesung Park
- **PoissonNMF**: Non-negative matrix factorization algorithm by Neher et al. (Biophysical Journal, 2009)
- **CSBDeep/CARE**: Probabilistic loss functions and uncertainty quantification (Schmidt & Weigert, BSD-3-Clause)
- **Reg-GAN**: Registration network components (Kong et al., NeurIPS 2021)

## Support

For questions and support:
1. Check the documentation in each subdirectory
2. Review the [THIRD_PARTY_ATTRIBUTION.md](THIRD_PARTY_ATTRIBUTION.md) for licensing details
3. Open an issue on GitHub
4. Contact the maintainers

---

**Note**: This software is designed for research purposes. For clinical or commercial applications, please ensure proper validation and compliance with relevant regulations.
