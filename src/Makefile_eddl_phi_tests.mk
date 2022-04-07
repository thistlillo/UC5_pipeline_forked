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
EXP_FLD = /opt/uc5/results/eddl_exp/eddl_ext_exp-eddl_ext_100_2000

$(EXP_FLD)/annotated_phi.tsv: $(CNN_MODEL_OUT_FN) $(REC_MODEL_OUT_FN) D01_gen_text_phi.py eddl_lib/text_generation_phi.py
	$(PYTHON) D01_gen_text_phi.py --out_fn=$@ --exp_fld=$(EXP_FLD) --img_fld=$(IMAGE_FLD)\
		--cnn_model=$(CNN_MODEL_OUT_FN) --rnn_model=$(REC_MODEL_OUT_FN) \
		--lstm_size=512 --emb_size=512 --n_tokens=$(MAX_TOKENS) \
		--tsv_file=$(IMG_BASED_DS_ENC) --img_size=$(CNN_IMAGE_SIZE) --dev=$(DEV_MODE_D)


test_generation:
	$(PYTHON) D01_gen_text_phi.py --out_fn=$@ --exp_fld=$(EXP_FLD) --img_fld=../data/image \
		-cnn_model=$(EXP_FLD)/cnn_sigmoid_eddl.onnx --rnn_model=$(EXP_FLD)/rec_checkpoint.bin \
		--lstm_size=512 --emb_size=512 --n_tokens=12 \
		--tsv_file=$(EXP_FLD)/img_reports_ext_enc.tsv --img_size=224 --nodev