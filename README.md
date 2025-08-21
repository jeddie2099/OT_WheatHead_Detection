# Improving wheat head detection with Optimal Transport-based domain adaptation

This project is derived from my master's thesis and shows the codes used to obtain the results for the *Domain Adaptation based on Optimal Transport to Enhance Wheat Head Detection via Deep Learning* paper. 

It consists of applying optimal transport (OT) techniques as preprocessing for the Global Wheat Head Detection (GWHD) 2021 images. These techniques, originally developed by *Feydy et al.* in the [GeomLoss](https://www.kernel-operations.io/geomloss/) library allow us to perform a color transfer that changes the color palette of the images, making them look more similar across the entire dataset. These images are then used to train an object detection model (YOLO), showing a better mAP on the test images than the same model trained on the images left unmodified. 

## Getting Started
This section contains the step-by-step instructions to set up and run this project locally. 

### Setting up the project folder
### Installing dependencies
### Applying OT-based domain adaptation
### Training and testing the models
### Inference with a single image

## File description
The repository contains the following python scripts and jupyter notebooks: 
- setup.py:
- utils.py:
- Domain_diversity_and_visual_OT.ipynb:
- OT_domain_adaptation.ipynb:
- Object_detection_models.ipynb:
- Inference.ipynb:

## Citation 
If you use this project, don't forget to cite the paper from which this repository derives, as well as the original GWHD 2021 dataset paper and the author of the original OT codes for color transfer:  

- E. Salas, G. Moreno, I. de la Rosa, D. Alaniz, J. Villa, and E. González, “Domain Adaptation based on Optimal Transport to Enhance Wheat Head Detection via Deep Learning”, [*Journal Name*], vol. XX, no. XX, pp. XX–XX, [year].. (paper is still being written)
- E. David, M. Serouart, D. Smith, S. Madec, K. Velumani, S. Liu, X. Wang, F. Pinto, S. Shafiee, I. Tahir, H. Tsujimoto, S. Nasuda, B. Zheng, N. Kirchgessner, H. Aasen, A. Hund, P. Sadeghi-Tehran, K. Nagasawa, G. Ishikawa, and W. Guo, [“Global Wheat Head Detection 2021: An Improved Dataset for Benchmarking Wheat Head Detection Methods”](https://www.sciencedirect.com/science/article/pii/S2643651524000591), *Plant Phenomics*, vol. 2021, pp. 1–9, Sep. 2021, doi: 10.34133/2021/9846158.
- J. Feydy, T. Séjourné, F.-X. Vialard, S.-i. Amari, A. Trouvé, and G. Peyré, [“Interpolating between Optimal Transport and MMD using Sinkhorn Divergences”](https://proceedings.mlr.press/v89/feydy19a/feydy19a.pdf), in *Proc. 22nd Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, 2019, pp. 2681–2690.
