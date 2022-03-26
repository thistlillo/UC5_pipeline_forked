THIS_MAKEFILE = $(lastword $(MAKEFILE_LIST))
$(warning running makefile: ${THIS_MAKEFILE})

EXP_NAME = eddl_acc_4gpu
CNN_IMAGE_SIZE = 224
SEED = 10
SHUFFLE_SEED = 40

BASE_DS_FLD = ../data
IMAGE_FLD = $(BASE_DS_FLD)/image
TEXT_FLD = $(BASE_DS_FLD)/text

PYTHON = python3
BASE_OUT_FLD = ../experiments_eddl

EXP_FLD  =$(BASE_OUT_FLD)/$(EXP_NAME)_$(SEED)
$(warning output folder for the experiment: ${EXP_FLD})

# TODO
$(EXP_FLD)/reports_raw.tsv:
preprocess:
	$(shell mkdir -p $(EXP_FLD))
	$(PYTHON) 00_preprocess.py --out_fld=$(EXP_FLD) --txt_fld=$(TEXT_FLD) --img_fld=$(IMAGE_FLD) \
		--min_term_freq_mesh=100 --min_term_freq_auto=100

#preprocess: $(EXP_FLD)/reports_raw.tsv

dataset:
	$(PYTHON) 01_train.py prepare_training --descr=$(EXP_NAME) --exp_fld=$(EXP_FLD) \
	--train_p=0.7 --valid_p=0.1 --shuffle_seed=$(SHUFFLE_SEED) --labels=mesh

train_cnn:
	@cp $(THIS_MAKEFILE) $(EXP_FLD)
	$(warning training cnn, folder: ${EXP_FLD})
	$(PYTHON) 01_train.py train_cnn --exp_fld=$(EXP_FLD) \
		--n_epochs=2000 --batch_size=[64,128] --learning_rate=[0.01,0.02,0.05,0.09] \
		--seed=$(SEED) --shuffle_seed=$(SHUFFLE_SEED) --verbose=True --dev=False \
		--gpu_id=[0,0,1,1] --description="resnet 101" \
		--remote_log=True

mean_iu-chest:
	$(PYTHON) utils/mean_and_variance.py --img_fld=$(IMAGE_FLD) --dataset=iu-chest

mean_chest-xray8: 
	$(PYTHON) utils/mean_and_variance.py --img_fld=/mnt/datasets/mimic-cxr/chestx-ray8/images --dataset=chest-xray8

mean_mimic_cxr:
	$(PYTHON) utils/mean_and_variance.py --in_tsv=/mnt/datasets/mimic-cxr/training_data/mimic/normal_bin_unbal.tsv --dataset=mimic_cxr

chest_xray8_normal:
	$(PYTHON) cnn_binary_train.py --in_fld=/mnt/datasets/uc5/UC5_pipeline_forked/experiments_eddl/chestxray_normal \
		--out_fld=/mnt/datasets/uc5/UC5_pipeline_forked/experiments_eddl/chestxray_normal \
		--gpu=[1,1] --nodev

chest_xray8_normal_new:
	$(PYTHON) cnn_binary_train.py --in_fld=/mnt/datasets/mimic-cxr/experiments/chest8 \
		--out_fld=/mnt/datasets/mimic-cxr/experiments/chest8 \
		--gpu=[1,1] --nodev

mimic_normal_unbal:
	$(PYTHON) cnn_binary_train.py --in_fld=/opt/uc5/results/eddl_exp/mimic/normal_unbal \
			--out_fld=/opt/uc5/results/eddl_exp/mimic/normal_unbal/out_accuracy \
			--dataset=mimic_cxr \
			--gpu=[1,1] --nodev



clean:
	rm -f $(EXP_FLD)/reports_raw.tsv

.PHONY: clean preprocess