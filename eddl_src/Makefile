THIS_MAKEFILE = $(lastword $(MAKEFILE_LIST))
$(warning running makefile: ${THIS_MAKEFILE})

EXP_NAME = eddl_integration
CNN_IMAGE_SIZE = 224
SEED = 999
SHUFFLE_SEED = 2

BASE_DS_FLD = ../data
IMAGE_FLD = $(BASE_DS_FLD)/image
TEXT_FLD = $(BASE_DS_FLD)/text

PYTHON = python3
BASE_OUT_FLD = ../experiments_eddl

EXP_FLD  =$(BASE_OUT_FLD)/$(EXP_NAME)_$(SEED)
$(warning output folder for the experiment: ${EXP_FLD})

$(EXP_FLD)/reports_raw.tsv:
preprocess:
	$(shell mkdir -p $(EXP_FLD))
	$(PYTHON) 00_preprocess.py --out_fld=$(EXP_FLD) --txt_fld=$(TEXT_FLD) --img_fld=$(IMAGE_FLD) \
		--min_term_freq_mesh=100 --min_term_freq_auto=100

#preprocess: $(EXP_FLD)/reports_raw.tsv

dataset:
	$(PYTHON) 01_train.py prepare_training --descr=$(EXP_NAME) --exp_fld=$(EXP_FLD) --train_p=0.7 --valid_p=0.1 --shuffle_seed=$(SHUFFLE_SEED) --labels=mesh

train_cnn:
	$(warning training cnn, folder: ${EXP_FLD})
	$(PYTHON) 01_train.py train_cnn --exp_fld=$(EXP_FLD) \
		--n_epochs=1000 --remote_log=True \
		--seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED)

clean:
	rm -f $(EXP_FLD)/reports_raw.tsv

.PHONY: clean preprocess