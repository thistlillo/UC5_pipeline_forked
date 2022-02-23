# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import fire
import eddl_lib.cnn_classification


# --------------------------------------------------
if __name__ == "__main__":
    fire.Fire(eddl_lib.cnn_classification.main)