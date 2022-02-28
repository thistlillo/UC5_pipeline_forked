THIS_MAKEFILE = $(lastword $(MAKEFILE_LIST))
$(warning running makefile: ${THIS_MAKEFILE})

EXP_NAME = eddl_integration
CNN_IMAGE_SIZE = 224
RANDOM_SEED = 999
SHUFFLE_SEED = 2

BASE_DS_FLD = ../data
IMAGE_FLD = $(BASE_DS_FLD)/image
TEXT_FLD = $(BASE_DS_FLD)/text

PYTHON = python3
BASE_OUT_FLD = ../experiments_eddl

EXP_FLD  =$(BASE_OUT_FLD)/$(EXP_NAME)_$(RANDOM_SEED)
$(warning output folder for the experiment: ${EXP_FLD})

$(EXP_FLD)/reports_raw.tsv:
preprocess:
	$(shell mkdir -p $(EXP_FLD))
	$(PYTHON) 00_preprocess.py --out_fld=$(EXP_FLD) --txt_fld=$(TEXT_FLD) --img_fld=$(IMAGE_FLD) \
		--min_term_freq_mesh=100 --min_term_freq_auto=100

#preprocess: $(EXP_FLD)/reports_raw.tsv


clean:
	rm -f $(EXP_FLD)/reports_raw.tsv

.PHONY: clean preprocess