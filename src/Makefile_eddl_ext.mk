#
# Project DeepHealth
# Use Case 5 (UC5)
# Deep Image Annotation
#
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
# A_pipeline:


# THIS_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
THIS_MAKEFILE = $(lastword $(MAKEFILE_LIST))
$(warning running makefile ${THIS_MAKEFILE})

PYTHON = python3

## -----------------------------------------------
LIBRARY = eddl
$(warning using library $(LIBRARY))

# -----------------------------------------------
# EXPERIMENT AND MODEL IDENTIFIERS
EXP_NAME = $(LIBRARY)_ext
MODEL = $(LIBRARY)_ext

# -----------------------------------------------
# FOLDERS & FILENAMES
BASE_DS_FLD = ../data
IMAGE_FLD = $(BASE_DS_FLD)/image
TEXT_FLD = $(BASE_DS_FLD)/text

## -----------------------------------------------
RANDOM_SEED = 100
SHUFFLE_SEED = 2000


# -----------------------------------------------
BASE_OUT_FLD = ../experiments_$(LIBRARY)
EXP_FLD = $(BASE_OUT_FLD)/$(MODEL)_exp-$(EXP_NAME)_$(RANDOM_SEED)_$(SHUFFLE_SEED)
$(shell mkdir -p $(EXP_FLD))

TSV_FLD = $(BASE_OUT_FLD)/tsv_$(LIBRARY)
RESULTS_FLD = $(EXP_FLD)/results

# prefix
REPORTS = reports_ext

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

PP_IMG_SIZE=300
PREPROC_IMAGES = $(TSV_FLD)/images_$(PP_IMG_SIZE).pickle

KEEP_N_TERMS = 100
# number of tags (MeSH term) to keep per report -- currently ignored
N_TERMS_PER_REP = 4
MIN_TERM_FREQ = 90
# number of images to keep per report, 0 = all
KEEP_N_IMGS = 0
#! image size (square) expected by the CNN
CNN_IMAGE_SIZE = 224

VERBOSITY_A = False

# -----------------------------------------------
# NOTICE: 
# - REPORTS_RAW_TSV is saved into $(TSV_FLD)
# - REPORTS_TSV is saved into $(EXP_FLD) since it applies experimentantion-specific filters on the raw values
$(REPORTS_RAW_TSV): A00_prepare_raw_tsv.py
	@mkdir -p $(TSV_FLD)
	$(PYTHON) A00_prepare_raw_tsv.py --txt_fld=$(TEXT_FLD) --img_fld=$(IMAGE_FLD) --out_file=$@ $(VERBOSITY_A) --stats

# -----------------------------------------------
$(REPORTS_TSV): $(REPORTS_RAW_TSV) A01_prepare_tsv.py
	@mkdir -p $(EXP_FLD)
	$(PYTHON) A01_prepare_tsv.py --out_file=$@ --raw_tsv=$(REPORTS_RAW_TSV) \
		--min_term_frequency=120 --n_terms=$(KEEP_N_TERMS) --n_terms_per_rep=$(N_TERMS_PER_REP) \
		--keep_no_indexing=True --keep_n_imgs=$(KEEP_N_IMGS) --verbose=$(VERBOSITY_A)

$(warning ${PREPROC_IMAGES})
$(PREPROC_IMAGES): 
	$(PYTHON) A02_preprocess_images.py --out_fn=$@ \ 
		--img_fld=$(IMAGE_FLD) --img_size=$(PP_IMG_SIZE) \
		--verbose=False 

# -----------------------------------------------
process_raw_dataset : $(REPORTS_RAW_TSV)

reports_tsv:  $(REPORTS_TSV)

preprocess_images: $(PREPROC_IMAGES)

A_pipeline: | process_raw_dataset reports_tsv

A_pipeline_clean:
	rm -f $(REPORTS_RAW_TSV) 
	rm -f $(REPORTS_TSV)
#rm -f $(PREPROC_IMAGES)
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

# -----------------------------------------------
VERBOSITY_B = False
$(REPORTS_CLEAN): $(REPORTS_TSV) B00_clean_text.py
	$(PYTHON) B00_clean_text.py --out_file=$@ --in_file=$(REPORTS_TSV) --out_fld=$(EXP_FLD) \
		--in_cols=['findings','impression'] --out_col=$(TEXT_COL) --verbose=$(VERBOSITY_B) --cleaning=v1

clean_text: $(REPORTS_CLEAN)

# -----------------------------------------------
# using a witness, not the best solution but clear enough. See https://www.gnu.org/software/automake/manual/html_node/Multiple-Outputs.html
# needed for using GNU make with version < 4.3. With GNU Make versions >= 4.3 it is possible to use "grouped targets".
ENC_WITNESS = $(EXP_FLD)/.enc_witness
$(ENC_WITNESS): $(REPORTS_CLEAN) B01_encode_data.py
	$(warning B01_encode.py produces multiple output files. Using witness ${ENC_WITNESS} in Makefile)
	@rm -f $@.tmp
	@touch $@.tmp
	$(PYTHON) B01_encode_data.py --in_file=$(REPORTS_CLEAN) --out_fld=$(EXP_FLD) --out_tsv=$(REPORTS_ENC) \
		--out_img_tsv=$(IMG_BASED_DS_ENC) --term_column=$(TERM_COLUMN) --text_column=$(TEXT_COL) \
		--vocab_size=$(VOCABULARY_SIZE) --min_freq=$(MIN_WORD_FREQ) --sen_len=$(MAX_SENTENCE_LENTH) $(VERBOSITY_B)
	@mv -f $@.tmp $@

encode: $(ENC_WITNESS)

# -----------------------------------------------
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

# -----------------------------------------------
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
# -----------------------------------------------
# softmax or sigmoid:
CNN_OUT_LAYER=sigmoid
CNN_MODEL_OUT_FN = $(EXP_FLD)/cnn_$(CNN_OUT_LAYER)_eddl.onnx
# the following without extension because several files are saved, with different exts
REC_MODEL_OUT_FN = $(EXP_FLD)/rnn_rec.onnx
# the following files are saved after training and testing by C01_2_rec...
REC_MODEL_OUT_PRED_FN = $(subst .onnx,_pred.onnx,$(REC_MODEL_OUT_FN))
REC_MODEL_OUT_FN_BIN = $(subst .onnx,.bin,$(REC_MODEL_OUT_FN))
REC_MODEL_OUT_PRED_FN_BIN = $(subst .onnx,_pred.bin,$(REC_MODEL_OUT_FN))

CHECK_VAL_EVERY_CNN=10
CHECK_VAL_EVERY_RNN=20

BATCH_SIZE = 32
EDDL_CS = gpu
EDDL_CS_MEM = full_mem
EMB_SIZE = 512
GPU_ID_CNN=[0,1,0,0]
GPU_ID_RNN=[0,1,0,0]

# LAST_BATCH in {drop, random}: if |last batch|<BATCH_SIZE, 
# 		drop -> do not use the examples, random->choose the missing example randomly 
LAST_BATCH = random
LR = 0.05
LSTM_SIZE = 512
MAX_SENTENCES = 5
MAX_TOKENS = 12
MOMENTUM = 0.9
N_EPOCHS=10000
OPTIMIZER=adam
PATIENCE_KICK_IN=200
# PATIENCE TRESH corresponds to the number of validation steps, i.e. epochs = PATIENCE_THRESH * CHECK_VAL_EVERY
PATIENCE_THRESH=4
TRAIN_PERCENTAGE = 0.7
VALIDATION_PERCENTAGE = 0.1

VERBOSITY_C = False
DEBUG_C = False
DEV_MODE_C = False

REMOTE_LOG_CNN = True
REMOTE_LOG_RNN = True

# C01_cnn_module.py
$(CNN_MODEL_OUT_FN): $(SPLIT_WITNESS) C01_1_cnn_mod_edll.py 
	$(warning training target is $@)
	$(PYTHON) C01_1_cnn_mod_edll.py train  --out_fn=$@ \
		--preload_images=True --preproc_images=$(PREPROC_IMAGES) \
		--cnn_out_layer=$(CNN_OUT_LAYER) \
		--in_tsv=$(IMG_BASED_DS_ENC) --exp_fld=$(EXP_FLD)  \
		--img_fld=$(IMAGE_FLD) --text_column=$(TEXT_COL) \
		--seed=$(RANDOM_SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=$(N_EPOCHS) --batch_size=$(BATCH_SIZE) --last_batch=$(LAST_BATCH) \
		--train_p=$(TRAIN_PERCENTAGE) --valid_p=$(VALIDATION_PERCENTAGE) \
		--lstm_size=512 --emb_size=512 \
		--n_tokens=$(MAX_TOKENS) --n_sentences=1 \
		--check_val_every=$(CHECK_VAL_EVERY_CNN) --patience=$(PATIENCE_THRESH) --patience_kick_in=$(PATIENCE_KICK_IN) \
		--optimizer=$(OPTIMIZER) --lr=$(LR) --momentum=$(MOMENTUM) \
		--eddl_cs=$(EDDL_CS) --eddl_cs_mem=$(EDDL_CS_MEM) --gpu_id=$(GPU_ID_CNN) \
		--verbose=$(VERBOSITY_C) --debug=$(DEBUG_C) --dev=$(DEV_MODE_C) --remote_log=$(REMOTE_LOG_CNN)

train_cnn: $(CNN_MODEL_OUT_FN)

train_cnn_clean:
	rm -f $(CNN_MODEL_OUT_FN)

# -----------------------------------------------
$(REC_MODEL_OUT_FN): $(CNN_MODEL_OUT_FN) C01_2_rec_mod_edll.py
	$(PYTHON)  C01_2_rec_mod_edll.py train --out_fn=$@ \
		--preload_images=True --preproc_images=$(PREPROC_IMAGES) \
		--cnn_file=$(CNN_MODEL_OUT_FN) \
		--in_tsv=$(IMG_BASED_DS_ENC) --exp_fld=$(EXP_FLD) \
		--img_fld=$(IMAGE_FLD) --term_column=$(TERM_COLUMN) --text_column=$(TEXT_COL) \
		--seed=$(RANDOM_SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=$(N_EPOCHS) --batch_size=$(BATCH_SIZE) --last_batch=$(LAST_BATCH) \
		--train_p=$(TRAIN_PERCENTAGE) --valid_p=$(VALIDATION_PERCENTAGE) \
		--lstm_size=512 --emb_size=512 --n_tokens=$(MAX_TOKENS) \
		--eddl_cs=$(EDDL_CS) --eddl_cs_mem=$(EDDL_CS_MEM) --gpu_id=$(GPU_ID_RNN) \
		--check_val_every=$(CHECK_VAL_EVERY_RNN) \
		--remote_log=$(REMOVE_LOG_RNN) \
		--verbose=$(VERBOSITY_C) --debug=$(DEBUG_C) --dev=$(DEV_MODE_C)
# $(DEV_MODE_C)

train_rec: $(REC_MODEL_OUT_FN)

train_rec_clean:
	rm -f $(REC_MODEL_OUT_FN)
	rm -f $(REC_MODEL_OUT_PRED_FN)
	rm -f $(REC_MODEL_OUT_FN_BIN)
	rm -f $(REC_MODEL_OUT_PRED_FN_BIN)
	
train: $(REC_MODEL_OUT_FN)

# -----------------------------------------------
C_pipeline: | split_data train

C_pipeline_clean: train_rec_clean
	$(warning check recipe)
	rm -f $(CNN_MODEL_OUT_FN)
	rm -f $(EXP_FLD)/cnn_checkpoint.onnx
	rm -f $(SPLIT_WITNESS)
	rm -f $(addprefix $(EXP_FLD)/, test_ids.txt train_ids.txt valid_ids.txt)
	rmdir $(EXP_FLD)

#
# *** P I P E L I N E   D ***
#
DEV_MODE_D = True

$(EXP_FLD)/annotated_phi.tsv: $(CNN_MODEL_OUT_FN) $(REC_MODEL_OUT_FN) D01_gen_text_phi.py eddl_lib/text_generation_phi.py
	$(PYTHON) D01_gen_text_phi.py --out_fn=$@ --exp_fld=$(EXP_FLD) --img_fld=$(IMAGE_FLD)\
		--cnn_model=$(CNN_MODEL_OUT_FN) --rnn_model=$(REC_MODEL_OUT_FN) \
		--lstm_size=512 --emb_size=512 --n_tokens=$(MAX_TOKENS) \
		--tsv_file=$(IMG_BASED_DS_ENC) --img_size=$(CNN_IMAGE_SIZE) --dev=$(DEV_MODE_D)

annotate_phi: $(EXP_FLD)/annotated_phi.tsv

annotate_phi_clean:
	rm -f $(EXP_FLD)/annotated_phi.tsv


$(EXP_FLD)/cnn_classes.tsv: D02_cnn_classification.py
	$(PYTHON) D02_cnn_classification.py --out_fn=$@ --exp_fld=$(EXP_FLD) --img_fld=$(IMAGE_FLD)\
		--cnn_model=$(EXP_FLD)/cnn.onnx --tsv_file=$(IMG_BASED_DS_ENC) --img_size=$(CNN_IMAGE_SIZE) --dev=False

cnn_classification: $(EXP_FLD)/cnn_classes.tsv

cnn_classification_clean:
	rm -f $(EXP_FLD)/cnn_classes.tsv

D_pipeline_clean: | annotate_phi_clean cnn_classification_clean

#
# *** EXTENSIONS   E ***
#

cnn_ext:
	$(PYTHON) E_cnn_ext.py --out_fn="cnn_ext.onnx" --exp_fld=$(EXP_FLD) --img_fld=$(IMAGE_FLD)\
			--tsv_file=$(IMG_BASED_DS_ENC) --img_size=$(CNN_IMAGE_SIZE) --dev=False

# *** ***
all: train_rec

# *** ***
clean : | A_pipeline_clean B_pipeline_clean C_pipeline_clean D_pipeline_clean
	rmdir $(EXP_FLD)

# ***  *** 
.PHONY: all clean download process_raw_dataset reports_tsv \
	A_pipeline A_pipeline_clean \
	clean_text encode B_pipeline B_pipeline_clean \
	split_data train_cnn train_rnn C_pipeline C_pipeline_clean \
	train_cnn_clean train_rnn_clean \
	D_pipeline_clean annotate_phi  annotate_phi_clean \
	cnn_classification cnn_classification_clean \
	preprocess_images