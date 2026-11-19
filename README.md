# Reading the Lungs or the File

**Testing shortcut learning in three class chest X ray classification**

Saod AlMarzooqi and Fahim Alfahim, The Pennsylvania State University, DS 340W, Group 21

This is the code for our DS 340W research project. We ask whether a model that performs very well on a widely-used chest X ray dataset is actually reading the lungs or simply identifying the origin of the image file. The full paper is is [`Final_Paper.pdf`](Final_Paper.pdf).

## Main findings

* We reproduced the first parent paper. We achieved 94.3% over five folds with our VGG19, which is the same as the 94.2% implied by the confusion matrix of that paper.
* The decision tree that never saw a single pixel and only read file properties achieved a score of 96.3%, higher than all of the networks we trained.
* The scores drop dramatically when the image source is fixed or when it is different during training and testing, and they drop to chance or below when the image source is different across training and testing.
* The models have no preference for the lungs, with only 23% of the Grad CAM heat within the lungs and 24% of the picture in the lungs.
* Our lung guided training shifted the evidence of the model to the lungs and increased the pneumonia recall in the hardest group, but did not improve the fair test.

## What each notebook does

| Notebook | What it does | Paper sections | Run time on a T4 GPU |
|---|---|---|---|
| `01_shortcut_tests.ipynb` | Locates the three collections, trains a file clue model, executes tests within and between collections, and quantifies the amount of Grad CAM heat that is within the lungs. | IV.C, V.B, V.C, V.D, Table V | about 50 to 70 minutes (our estimate) |
| `02_reproduce_parent_paper.ipynb` | Reproduces the first parent paper with five frozen ImageNet networks, runs the lungs only input from the second parent paper, trains the file properties model, and fine tunes VGG19 | IV.B, V.A, V.B, Table IV | not timed |
| `03_lung_guided_training.ipynb` | Our first proposed novel approach: a source balanced weights and the lung guided loss on frozen VGG19 and DenseNet201, which are evaluated on an unseen set | III, V.E, Table II, Table VI, Fig. 1 | about 42 minutes |
| `04_lung_guided_fine_tuning.ipynb` | Our novel approach round 2: fine tunes the last block of VGG19 with a stricter penalty, and compares Grad CAM with attribution maps. | III.F, V.E, Table II, Fig. 2 | about 47 minutes |

## How to run

All notebooks run on Google Colab, and nothing needs to be installed by hand.

1. Download a Notebook from the repository
2. In Colab, choose **Runtime**, then **Change runtime type**, and make sure **T4 GPU** is selected.
3. Choose **Runtime**, then **Run all**.
4. Wait until the last cell finishes. Every table and figure appears inside the notebook, and the last section gathers the main results in one place.

Each notebook does the setup by itself:
* It downloads the dataset (version 2 of the Mendeley dataset, 6,939 images) directly from Mendeley Data.
* It clones our helper repository, which holds `lung_focus.py`, the script that runs the lung finder.
* `lung_focus.py` downloads the released lung finder model and weights of Bassi and Attux from their GitHub repository and checks both downloaded files with SHA256 hashes before using them.
* It writes its own small Python package (`xray_platform`) into the Colab session, which is why some cells print `Writing /content/xray_platform/...`.

## Numbers to check after a run

These are the main numbers in the paper and where they come from. Each one is printed in the outputs of its notebook.

| Notebook | Result | Value in the paper |
|---|---|---|
| 01 | File clue model and the best network on the three class task, grouped split | 94.7% and 94.7% |
| 01 | Normal against pneumonia AUC inside the PNG collection (networks) | 0.656 to 0.751 |
| 01 | Same test across the two collections | 0.265 to 0.561 |
| 01 | Grad CAM heat inside the lungs against the share of the picture the lungs fill | 23% against 24% |
| 02 | VGG19, ResNet50, InceptionV3, and DenseNet201 accuracy over five folds | 94.3%, 94.7%, 90.4%, 94.6% |
| 02 | DenseNet201 with the lungs only input | 90.6% |
| 02 | File properties model, no pixels | 96.3% |
| 02 | Fine tuned VGG19 on fold 1 | 94.8% |
| 03 | File properties model on the unseen set | 95.9% |
| 03 | Frozen VGG19 baseline on the unseen set: accuracy and PNG AUC | 94.4% and 0.656 |
| 03 | Frozen VGG19 with lung guidance: heat inside the lungs and PNG AUC | 56% and 0.586 |
| 04 | Fine tuned VGG19 baseline on the unseen set | 95.3% |
| 04 | Fine tuned VGG19 with lung guidance: PNG AUC and attribution inside the lungs | 0.685 and 71% |

The random seeds are fixed, but training on a GPU is not necessarily repeatable. The last digit may vary slightly from one new run to another, but the general pattern remains the same.

## Data

We use the COVID19, pneumonia and normal chest X ray PA dataset by Asraf and Islam, version 2, on Mendeley Data: [https://data.mendeley.com/datasets/mxc6vb7svm/2](https://data.mendeley.com/datasets/mxc6vb7svm/2). It has 2,313 images in each of the three classes. The images are not stored in this repository, because every notebook downloads them.

## Code from other sources

* **Helper repository:** [MarzSaod/pyimagesearch-covid19-image-classification](https://github.com/MarzSaod/pyimagesearch-covid19-image-classification) is our fork of an open Keras starter project. We added `lung_focus.py` and `report_utils.py` there, and the notebooks clone it automatically
* **Lung finder:** the trained lung segmentation network and its weights come from Bassi and Attux: [PedroRASB/COVID-19-detection-with-lung-segmentation](https://github.com/PedroRASB/COVID-19-detection-with-lung-segmentation).

## Papers we build on

1. H. Aljuaid et al., "An experimental comparison of deep learning models for pneumonia classification from chest X ray images," Biomedical Signal Processing and Control, 2026. First parent paper. [Link](https://www.sciencedirect.com/science/article/pii/S1746809425012534)
2. M. Kabak, M. T. Baran, and B. Çil, "Application of deep learning methods in the classification of normal and pneumonia lung images," Frontiers in Medicine, 2026. [Link](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1755871/full)
3. P. R. A. S. Bassi and R. Attux, "Covid 19 detection using chest X rays: Is lung segmentation important for generalization?" Research on Biomedical Engineering, 2022. Second parent paper. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC9628459)

The full reference list is in the paper.

## Archive

The `Notebooks` folder keeps our early notebooks from the first weeks of the course, when we were setting up the parent paper code and exploring the data. They are not needed to reproduce any result in the paper.
