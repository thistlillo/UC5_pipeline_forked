#
# Project DeepHealth
# Use Case 5 (UC5)
# Deep Image Annotation
#
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#


# THIS_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))
$(warning running makefile ${THIS_MAKEFILE})

PYTHON = python3
RANDOM_SEED = 100
SHUFFLE_SEED = 2000

# EXPERIMENT AND MODEL IDENTIFIERS
EXP_NAME = torch_std
MODEL = torch_std

# FOLDERS & FILENAMES
BASE_DS_FLD = ../data
IMAGE_FLD = $(BASE_DS_FLD)/image
TEXT_FLD = $(BASE_DS_FLD)/text

REPORTS = reports

BASE_OUT_FLD = ../experiments_torch
EXP_FLD = $(BASE_OUT_FLD)/$(MODEL)_exp-$(EXP_NAME)_$(RANDOM_SEED)_$(SHUFFLE_SEED)
$(shell mkdir -p $(EXP_FLD))

TSV_FLD = $(BASE_OUT_FLD)/tsv_torch
RESULTS_FLD = $(EXP_FLD)/results



# *** *** D O W N L O A D
$(BASE_DS_FLD)/NLMCXR_png.tgz: 
	cd $(BASE_DS_FLD) && wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz 

$(BASE_DS_FLD)/NLMCXR_reports.tgz:
	cd $(BASE_DS_FLD) && wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz

$(BASE_DS_FLD)/text: $(BASE_DS_FLD)/NLMCXR_reports.tgz
	cd $(BASE_DS_FLD) && tar xf NLMCXR_reports.tgz && mv ecgen-radiology text

$(BASE_DS_FLD)/image: $(BASE_DS_FLD)/NLMCXR_png.tgz
	cd $(BASE_DS_FLD) && mkdir image_ && mv NLMCXR_png.tgz image_ && cd image_ && tar xf NLMCXR_png.tgz && mv NLMCXR_png.tgz .. && cd .. && mv image_ image

download: | $(BASE_DS_FLD)/text $(BASE_DS_FLD)/image

#
# *** P I P E L I N E   A ***
#
REPORTS_RAW_TSV = $(TSV_FLD)/$(REPORTS)_raw.tsv
REPORTS_TSV = $(EXP_FLD)/$(REPORTS).tsv

KEEP_N_TERMS = 100
# number of tags (MeSH term) to keep per report
N_TERMS_PER_REP = 4
# number of images to keep per report, 0 = all
KEEP_N_IMGS = 0
#! image size (square) expected by the CNN
PREPROC_IMG_SIZE = 224

VERBOSITY_A = False

# NOTICE: 
# - REPORTS_RAW_TSV is saved into $(TSV_FLD)
# - REPORTS_TSV is saved into $(EXP_FLD) since it applies experimentantion-specific filters on the raw values
$(REPORTS_RAW_TSV): A00_prepare_raw_tsv.py
	$(shell mkdir -p $(TSV_FLD))
	@cp $(THIS_MAKEFILE) $(EXP_FLD)
	$(PYTHON) A00_prepare_raw_tsv.py --txt_fld=$(TEXT_FLD) --img_fld=$(IMAGE_FLD) --out_file=$@ $(VERBOSITY_A) --stats

$(EXP_FLD)/$(REPORTS).tsv: $(REPORTS_RAW_TSV) A01_prepare_tsv.py
	$(shell mkdir -p $(EXP_FLD))
	$(PYTHON) A01_prepare_tsv.py --out_file=$@ --raw_tsv=$(REPORTS_RAW_TSV) \
		--n_terms=$(KEEP_N_TERMS) --n_terms_per_rep=$(N_TERMS_PER_REP) \
		--keep_no_indexing=True --keep_n_imgs=$(KEEP_N_IMGS) --verbose=$(VERBOSITY_A)

process_raw_dataset : $(REPORTS_RAW_TSV)

reports_tsv:  $(REPORTS_TSV)

A_pipeline: | process_raw_dataset reports_tsv

A_pipeline_clean:
	rm -f $(REPORTS_RAW_TSV) 
	rm -f $(REPORTS_TSV)
	rmdir $(TSV_FLD)
# not removing EXP_FLD

#
# *** P I P E L I N E   B ***
#
# output files
REPORTS_CLEAN = $(EXP_FLD)/$(REPORTS)_clean.tsv
REPORTS_ENC = $(EXP_FLD)/$(REPORTS)_enc.tsv
# same as reports_enc, but with one row per image
IMG_BASED_DS_ENC = $(EXP_FLD)/img_$(REPORTS)_enc.tsv
# pipeline B produces other output files, whose name cannot be specified here:
#    - (EXP_FLD)/vocab.pickle : the vocabulary
#    - (EXP_FLD)/lab2index.json: term (from TERM_COLUMN, see below) to integer index
#    - (EXP_FLD)/index2lab.json: integer index to term, the reverse of the file above

# column to reads terms from (not relevant when using the base_model because TERM_COLUMN is not used)
TERM_COLUMN = auto_term

# (new, output) column storing the cleaned text
TEXT_COL = text
VOCABULARY_SIZE = 1000
MIN_WORD_FREQ = 2
MAX_SENTENCE_LENTH = 12

# IMG_SIZE: input to convolutional neural network
IMG_SIZE = 224

VERBOSITY_B = False
$(REPORTS_CLEAN): $(REPORTS_TSV) B00_clean_text.py
	$(PYTHON) B00_clean_text.py --out_file=$@ --in_file=$(REPORTS_TSV) --out_fld=$(EXP_FLD) \
		--in_cols=['findings','impression'] --out_col=$(TEXT_COL) --verbose=$(VERBOSITY_B) --cleaning=v1

clean_text: $(REPORTS_CLEAN)

# using a witness, not the best solution but clear enough. See https://www.gnu.org/software/automake/manual/html_node/Multiple-Outputs.html
# needed for using GNU make with version < 4.3. With GNU Make versions >= 4.3 it is possible to use "grouped targets".
ENC_WITNESS = $(EXP_FLD)/.enc_witness
$(ENC_WITNESS): $(REPORTS_CLEAN) B01_encode_data.py
	$(warning B01_encode.py produces multiple output files. Using witness ${ENC_WITNESS} in Makefile)
	@rm -f $@.tmp
	@touch $@.tmp
	$(PYTHON) B01_encode_data.py --in_file=$(REPORTS_CLEAN) --out_fld=$(EXP_FLD) --out_tsv=$(REPORTS_ENC) --out_img_tsv=$(IMG_BASED_DS_ENC) --term_column=$(TERM_COLUMN) --text_column=$(TEXT_COL) --vocab_size=$(VOCABULARY_SIZE) --min_freq=$(MIN_WORD_FREQ) --sen_len=$(MAX_SENTENCE_LENTH) $(VERBOSITY_B)
	@mv -f $@.tmp $@

encode: $(ENC_WITNESS)

B_pipeline: | clean_text encode

B_pipeline_clean:
	rm -f $(REPORTS_CLEAN)
	rm -f $(REPORTS_ENC)
	rm -f $(EXP_FLD)/vocab.pickle
	rm -f $(EXP_FLD)/index2lab.json
	rm -f $(EXP_FLD)/lab2index.json
	rm -f $(ENC_WITNESS)
	rm -f $(IMG_BASED_DS_ENC)

# rm -f $(EXP_FLD)/images_$(IMG_SIZE).pickle

#
# *** P I P E L I N E   C ***
#
# parameters
N_EPOCHS=10000
BATCH_SIZE = 32
TRAIN_PERCENTAGE = 0.7
VALIDATION_PERCENTAGE = 0.1
# LAST_BATCH in {drop, random}: if |last batch|<BATCH_SIZE, drop -> do not use the examples, random->choose the missing example randomly 
LAST_BATCH = drop

# model
EMB_SIZE = 512
LSTM_SIZE = 512
# single_channel_cnn: if set, the cnn will receive 1-channel images. Otherwise, 3-channel (RGB) images
#   The three channels are not averaged, but only the first (red:0) channel is kept.
# When using a pre-trained network, subsequent training has not been tested: the 3-channel input layer is substituted by a new one
SINGLE_CHANNEL_CNN = True

# text related
MAX_TOKENS = 12
MAX_SENTENCES = 5

VERBOSITY_C = False
DEBUG_C = False
DEV_MODE_C = True
GPU_ID = 1
REMOTE_LOG=True

SPLIT_WITNESS = $(EXP_FLD)/.split_witness
$(SPLIT_WITNESS): $(ENC_WITNESS) C00_split.py
	$(warning C00_split.py produces multiple output files. Using witness ${SPLIT_WITNESS} in Makefile)
	@rm -f $@.tmp
	@touch $@.tmp
	$(PYTHON) C00_split.py --in_tsv=$(IMG_BASED_DS_ENC) --exp_fld=$(EXP_FLD) \
		--term_column=$(TERM_COLUMN) --shuffle_seed=$(SHUFFLE_SEED) \
		--train_p=$(TRAIN_PERCENTAGE) --valid_p=$(VALIDATION_PERCENTAGE)
	@mv -f $@.tmp $@

split_data: $(SPLIT_WITNESS)

TORCH_OUT_FN=$(EXP_FLD)/model.pt
$(TORCH_OUT_FN): $(ENC_WITNESS) C01_train_torch.py
	$(PYTHON) C01_train_torch.py --out_fn=$@  \
	--only_images=False --load_data_split=False \
	--in_tsv=$(IMG_BASED_DS_ENC) --exp_fld=$(EXP_FLD)  --img_fld=$(IMAGE_FLD) \
	--term_column=$(TERM_COLUMN) --text_column=$(TEXT_COL) --seed=$(RANDOM_SEED) \
	--shuffle_seed=$(SHUFFLE_SEED) --n_epochs=$(N_EPOCHS) --batch_size=$(BATCH_SIZE) \
	--last_batch=$(LAST_BATCH) --train_p=$(TRAIN_PERCENTAGE) --valid_p=$(VALIDATION_PERCENTAGE) \
	--lstm_size=$(EMB_SIZE) --emb_size=$(EMB_SIZE) --text_column=$(TEXT_COL) --n_tokens=$(MAX_TOKENS) \
	--device=gpu --gpu_id=0 \
	--loader_threads=1 \
	--check_val_every=50 \
	--single_channel_cnn=True \
	--verbose=$(VERBOSITY_C) --debug=$(DEBUG_C) --dev=$(DEV_MODE_C) --remote_log=$(REMOTE_LOG)

train: $(TORCH_OUT_FN)

C_pipeline: | split_data train

C_pipeline_clean:
	rm -f $(TORCH_OUT_FN)
	rm -f $(EXP_FLD)/validation_best.pt
	rm -f $(EXP_FLD)/$(THIS_MAKEFILE)
	rm -f $(EXP_FLD)/checkpoint_e*_l*.pt
	rm -f $(SPLIT_WITNESS)
	rm -f $(addprefix $(EXP_FLD)/, test_ids.txt train_ids.txt valid_ids.txt)
	rmdir $(EXP_FLD)

# *** ***
all: train

# *** ***
clean : | A_pipeline_clean B_pipeline_clean C_pipeline_clean

# ***  *** 
.PHONY: all clean download process_raw_dataset reports_tsv \
	A_pipeline A_pipeline_clean \
	clean_text encode B_pipeline B_pipeline_clean \
	split_data train C_pipeline C_pipeline_clean
