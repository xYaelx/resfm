<div align="center">
<h1>RESfM: Robust Deep Equivariant Structure from Motion</h1>


**[Weizmann Institute of Science](https://www.weizmann.ac.il/pages/)**; **[NVIDIA Research](https://research.nvidia.com/home)**


[Fadi Khatib](https://fadikhatib.github.io/), [Yoni Kasten](https://ykasten.github.io/), [Dror Moran](https://scholar.google.com/citations?user=kS5jfSoAAAAJ&hl=en), [Meirav Galun](https://www.weizmann.ac.il/math/meirav/), [Ronen Basri](https://www.weizmann.ac.il/math/ronen/home)
</div>

```bibtex
@inproceedings{khatib2025resfm,
  title={Resfm: Robust deep equivariant structure from motion},
  author={Khatib, Fadi and Kasten, Yoni and Moran, Dror and Galun, Meirav and Basri, Ronen},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```


## Overview

Robust Deep Equivariant Structure from Motion (RESfM, ICLR 2025) is a deep-learning framework for multiview SfM that overcomes the assumption of clean point tracks. Unlike prior equivariant architectures, RESfM integrates a learnable inlier/outlier classifier into a row-and-column permutation‑equivariant backbone. This design enables it to accurately recover camera poses and 3D structure from noisy, outlier‑contaminated tracks, achieving results competitive with classical methods like COLMAP while being significantly faster. The method generalizes across large datasets with hundreds of views.


## Quick Start

First, clone this repository to your local machine, and install the dependencies (torch, torchvision, numpy, Pillow, and huggingface_hub). 

```bash
git clone https://github.com/FadiKhatib/resfm.git  
cd resfm
pip install -r requirements.txt
```

## Detailed Usage

<details>
<summary>Click to expand</summary>



## Exporting to COLMAP Format



The reconstruction result (camera parameters and 3D points) will be automatically saved in the COLMAP format.


## Integration with Gaussian Splatting


The exported COLMAP files can be directly used with [gsplat](https://github.com/nerfstudio-project/gsplat) for Gaussian Splatting training



## Acknowledgements

Thanks to these great repositories: [ESFM](https://github.com/drormoran/Equivariant-SFM), [GASFM](https://github.com/lucasbrynte/gasfm), [Kornia](https://github.com/kornia/kornia), [PyCOLMAP](https://colmap.github.io/pycolmap/index.html)

## Checklist

- [ ] Release all the datasets
- [ ] Release the code for creating tracks from images


## License
See the [LICENSE](./LICENSE.txt) file for details about the license under which this code is made available.

