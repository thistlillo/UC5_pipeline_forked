# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import os
from posixpath import join

from pytorch_lightning.loggers import NeptuneLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch

from lightning.uc5_data_module import Uc5DataModule
from lightning.uc5_model_lightning import Uc5Model

class Uc5Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.logger = self._configure_logger()
    #<

    def _configure_logger(self):
        mode="async"

        if self.conf["debug"] or self.conf["dev"]:
            mode = "debug"
        
        if not self.conf["remote_log"]:
            mode = "offline"

        neptune_logger = NeptuneLogger(
            project = "thistlillo/UC5-DeepHealth-PT_Lightning",
            name = "base_version",
            mode = mode
        )
        return neptune_logger
    #<

    def train(self):
        model = Uc5Model(self.conf)
        data_module = Uc5DataModule(self.conf)
        
        check_val_every_n_epoch = self.conf["check_val_every_n_epoch"]
        #> dev mode
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0  # limit_train_batches
        if self.conf["dev"]:
            limit_train_batches = 0.05
            limit_val_batches = 0.05
            limit_test_batches = 0.05
            self.conf["n_epochs"] = 1
            check_val_every_n_epoch = 1
        
        #<
        
        checkpoint_fld = join(self.conf["exp_fld"], "checkpoints")
        os.makedirs(checkpoint_fld, exist_ok=True)
        
        # checkpoints, best model
        checkpoint_callback = ModelCheckpoint(
           monitor="val/batch/val_loss",
           dirpath=checkpoint_fld,
           filename="uc5model-{epoch:02d}-{val_loss:.2f}",
           save_top_k=1,
           mode="min",
           )

        #> device
        params = {"accelerator": self.conf["accelerator"],
                "gpus": self.conf["gpus"],
        }
        #< 

        #> automatic mixed precision
        if self.conf["amp_backend"]:
            params["amp_backend"] = self.conf["amp_backend"]
            params["amp_level"] = self.conf["amp_level"]
            params["precision"] = self.conf["precision"]

        if self.conf["strategy"]:
            params["strategy"] = self.conf["strategy"]
        #<

        #> lightning trainer
        trainer = pl.Trainer(
           **params,
            limit_train_batches = limit_train_batches, limit_val_batches = limit_val_batches, limit_test_batches = limit_test_batches,
            logger = self.logger,
            max_epochs = self.conf["n_epochs"],
            callbacks = [checkpoint_callback],
            check_val_every_n_epoch=check_val_every_n_epoch,
        )
        #<

        #> training
        print("training starting - calling fit(.)")
        
        trainer.fit(model, datamodule=data_module)
        print(f"best model at: {checkpoint_callback.best_model_path}")
        print(checkpoint_callback.best_model_path)
        # save the name of the best model since we insert epoch and loss in the filename
        with open( join(checkpoint_fld, "bestmodel.txt"), "w") as fout:
            fout.write(checkpoint_callback.best_model_path)
        print("training completed")
        #<

        #> test
        print("test 1/2: testing model at last epoch- calling test(.)")
        trainer.test(model, datamodule=data_module)
        
        torch.save(model.state_dict(), self.conf["out_fn"])
        print(f"model at last epoch saved: {self.conf['out_fn']}")

        print("test 2/2: testing best model on the validation set - calling test(.)")
        model = Uc5Model.load_from_checkpoint(checkpoint_callback.best_model_path, conf=self.conf)
        trainer.test(model, datamodule=data_module)

        best_on_val_fn = join(self.conf["exp_fld"], "validation_best.pt")
        torch.save(model.state_dict(), best_on_val_fn)
        print(f"best model on validation saved in pytorch format at: {best_on_val_fn}")
        #<
        
        print("test completed")
        print("done.")
    #< train
#< class