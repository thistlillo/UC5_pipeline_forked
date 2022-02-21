## UC5 "Deep Image Annotation", DeepHealth

Repository for the Use Case 5 "Deep Image Annotation"

The UC5 pipeline is composed by three stages A, B and C. The filename of the Python scripts starts with the letter of the pipeline they belong to, followed by the step number.
- A_pipeline:
  - `A00_prepare_raw_tsv.py`: processes the raw dataset and organize the information in a .tsv file;
  - `A01_prepare_tsv.py`: filters the data that will be used for building the model (e.g, it excluded reports without images).
- B_pipeline:
  - `B00_clean_text.py`: text cleaning;
  - `B01_encode_data.py`: data encoding for training the model.
- C_pipeline:
  - `C00_split.py`: splits the data into training, validation and test set;
  - `C01_train[torch|lightning|eddl].py`: trains, validates and tests the model using the three libraries.
- D_pipeline: generates the text. Utility script for loading trained models and generating text for an image or the full dataset (instead of word indexes as in pipeline C).
  - `D01_gen_text_phi.py`: generates text for the full dataset (or a single image). 


The repository contains three implementations of the UC5 model based on the three deep learning libraries:
- **EDDLL**
- **PyTorch**
- **PyTorch-Lightning**

With the PyTorch-Lightning library it is relatively easy to train and test the model on multiple GPUs, multiple nodes, and with different level of precision or automatic mixed precision.

Pipelines A and B are shared among the three implementations. Library-specific files are organized in the subfolders of `src/eddl`, `src/pt` and `src/lightning` for, respectively, EDDL, PyTorch and PyTorch-Lightning.

Each implementation is "managed" via a Makefile. In the `src` folder you may find three different Makefile files, whose suffixes indicate  the corresponding implementation. For example, the makefile for the EDDL implementation is named `Makefile_eddl.mk` (the current EDDL pipeline is managed by `Makefile_eddl_phi.mk`). The makefiles are to meant as a way to manage the dataflow and make the experiments more easily reproducible: they are not to be interpreted as C makefiles. Indeed, many steps do not have an explicity dependency on the main Python file that generates the output (target): for example, there are no dependencies between the output files for trained models and the scripts performing the training (to avoid starting a long training procedure, corrupting files, as a consequence of minor fixes in the source).

All of the implementations expects as input the dataset Indiana University Chest X-Ray Collection:
- dataset available here: https://openi.nlm.nih.gov/faq
- publication describing the dataset: https://pubmed.ncbi.nlm.nih.gov/26133894/
The file `radiology_vocabulary_final.xlsx`, now part of the dataset, has been published only recently and is not used in the current implementation. The dataset may be downloaded using the Makefiles provided in this repository.

## Instructions
Set up the environment installing the correct libraries. You may find two requirement files for creating a conda environment:
- `torch_lightning_reqs.txt`: for the PyTorch and the PyTorch-Lightning pipelines
- `eddl_reqs.txt`: for the EDDL pipeline (not yet published)

In order to run an experiment:
```
make -f Makefile_[eddl|torch|lightning] download
make -f Makefile_[eddl|torch|lightning] all
```
**If you decide to download the dataset via the makefile, it strongly advised to check the paths specified in the makefile.**

**IMPORTANT**
Experiments are tracked remotely using the platform Neptune.ai (https://neptune.ai/).
If you do not have an account, please set the variable REMOTE_LOG to False in the makefile.

**NOTE**
The makefile (.mk) is automatically copied in the output folder of the experiment.
