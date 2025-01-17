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
EXP_FLD = /opt/uc5/results/eddl_exp/wp6_without_normal
IMAGE_FLD = /mnt/datasets/uc5/std-dataset/image
TERM_COLUMN = auto_term
RANDOM_SEED = 100
SHUFFLE_SEED = 2000

BATCH_SIZE = 64
EDDL_CS = gpu
EDDL_CS_MEM = full_mem
EMB_SIZE = 512
GPU_ID_CNN=[1,1,0,0]
GPU_ID_RNN=[0,0,1,1]


TRAIN_PERCENTAGE = 0.7
VALIDATION_PERCENTAGE = 0.1
MAX_TOKENS = 12
# (new, output) column storing the cleaned text
TEXT_COL = text

$(EXP_FLD)/annotated_phi.tsv: $(CNN_MODEL_OUT_FN) $(REC_MODEL_OUT_FN) D01_gen_text_phi.py eddl_lib/text_generation_phi.py
	$(PYTHON) D01_gen_text_phi.py --out_fn=$@ --exp_fld=$(EXP_FLD) --img_fld=$(IMAGE_FLD)\
		--cnn_model=$(CNN_MODEL_OUT_FN) --rnn_model=$(REC_MODEL_OUT_FN) \
		--lstm_size=512 --emb_size=512 --n_tokens=$(MAX_TOKENS) \
		--tsv_file=$(IMG_BASED_DS_ENC) --img_size=$(CNN_IMAGE_SIZE) --dev=$(DEV_MODE_D)

CNN_MODEL_OUT_FN = $(EXP_FLD)/best_cnn.onnx

$(EXP_FLD)/recurrent.onnx:  C01_2_rec_mod_edll.py
	$(PYTHON)  C01_2_rec_mod_edll.py train --out_fn=$@ \
		--preload_images=False --preproc_images=$(PREPROC_IMAGES) \
		--cnn_file=$(CNN_MODEL_OUT_FN) \
		--in_tsv=$(EXP_FLD)/img_reports.tsv --exp_fld=$(EXP_FLD) \
		--img_fld=$(IMAGE_FLD) --term_column=$(TERM_COLUMN) --text_column=$(TEXT_COL) \
		--seed=$(RANDOM_SEED) --shuffle_seed=$(SHUFFLE_SEED) \
		--n_epochs=5000 --batch_size=$(BATCH_SIZE) --last_batch=drop \
		--train_p=$(TRAIN_PERCENTAGE) --valid_p=$(VALIDATION_PERCENTAGE) \
		--lstm_size=512 --emb_size=512 --n_tokens=$(MAX_TOKENS) \
		--eddl_cs=$(EDDL_CS) --eddl_cs_mem=$(EDDL_CS_MEM) --gpu_id=$(GPU_ID_RNN) \
		--check_val_every=2\
		--remote_log=True \
		--verbose=True --nodebug --nodev \
		--label_col=auto_labels

train_rec : $(EXP_FLD)/recurrent.onnx

GEN_FLD = /opt/uc5/results/eddl_exp/wp6_noatt_TRAINED
test_generation:
	$(PYTHON) D01_gen_text_phi.py --out_fn=$@ --exp_fld=$(GEN_FLD) --img_fld=../data/image \
		-cnn_model=$(GEN_FLD)/cnn_84val_neptune179.onnx --rnn_model=$(EXP_FLD)/recurrent_best_train.bin \
		--lstm_size=512 --emb_size=512 --n_tokens=12 \
		--tsv_file=$(GEN_FLD)/img_reports_phi2_enc.tsv --img_size=224 --nodev


test_generation_2.tsv:
	$(PYTHON) D01_gen_text_phi_2.py --out_fn=$@ --exp_fld=$(GEN_FLD) --img_fld=../data/image \
		-cnn_model=$(GEN_FLD)/cnn_84val_neptune179.onnx --rnn_model=$(GEN_FLD)/recurrent_best_train.bin \
		--lstm_size=512 --emb_size=512 --n_tokens=12 \
		--tsv_file=$(GEN_FLD)/img_reports_phi2_enc.tsv --img_size=224 --nodev

test_generation_3:
	$(PYTHON) D01_gen_text_phi_3.py --out_fn=$@ --exp_fld=$(GEN_FLD) --img_fld=../data/image \
		-cnn_model=$(GEN_FLD)/cnn_84val_neptune179.onnx --rnn_model=$(GEN_FLD)/recurrent_best_valid.bin \
		--lstm_size=512 --emb_size=512 --n_tokens=12 \
		--tsv_file=$(GEN_FLD)/img_reports_phi2_enc.tsv --img_size=224 --nodev


test_generation2: test_generation_2.tsv