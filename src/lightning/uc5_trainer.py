from posixpath import join

from pytorch_lightning.loggers import NeptuneLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pt.uc5_data_module import Uc5DataModule
from pt.uc5_model_lightning import Uc5Model

class Uc5Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.logger = self._configure_logger()
        # self.data_module = self._load_data_module()
        #self.model = self._load_model()

    def _configure_logger(self):
        mode="async"

        if self.conf["debug"] or self.conf["dev"]:
            mode = "debug"
        
        if not self.conf["remote_log"]:
            mode = "offline"

        #> section: neptune logger, remember to set environment variable
        neptune_logger = NeptuneLogger(
            project = "thistlillo/UC5-DeepHealth-PT_Lightning",
            name = "base_version",
            mode = mode
        )
        #< section: neptune logger, remember to set environment variable end
        return neptune_logger

    def _load_data_module(self):
        dm = Uc5DataModule(self.conf)
        return dm

    def _load_model(self):
        uc5model = Uc5Model(self.conf)
        return uc5model

    def train(self):
        model = self._load_model()
        data_module = self._load_data_module()

        # dev mode >
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0  # limit_train_batches
        if self.conf["dev"]:
            limit_train_batches = 0.05
            limit_val_batches = 0.1
            limit_test_batches = 0.1
            self.conf["n_epochs"] = 1
        
        
        
        # <
        
        checkpoint_fld = join(self.conf["exp_fld"], "checkpoints")
        # checkpoints, best model
        checkpoint_callback = ModelCheckpoint(
           monitor="avg_val_loss",
           dirpath=checkpoint_fld,
           filename="uc5model-{epoch:02d}-{val_loss:.2f}",
           save_top_k=1,
           mode="min",
           )


        params = {"accelerator": self.conf["accelerator"],
                "gpus": self.conf["gpus"],
        }
        if self.conf["amp_backend"]:
            params["amp_backend"] = self.conf["amp_backend"]
            params["amp_level"] = self.conf["amp_level"]
            params["precision"] = 16

        if self.conf["strategy"]:
            params["strategy"] = self.conf["strategy"]

        # trainer >
        trainer = pl.Trainer(
           **params,
            # accelerator = "dpp",
            # precision = 16,
            limit_train_batches = limit_train_batches, limit_val_batches = limit_val_batches, limit_test_batches = limit_test_batches,
            logger = self.logger,
            max_epochs = self.conf["n_epochs"],
            callbacks = [checkpoint_callback],
            check_val_every_n_epoch=25,
        )
        # <

        print("training starting - calling fit(.)")
        trainer.fit(model, datamodule=data_module)
        print(f"best model at: {checkpoint_callback.best_model_path}")
        with open( join(checkpoint_fld, "bestmodel.txt"), "w") as fout:
            fout.write(checkpoint_callback.best_model_path)
    
        print("training completed, now testing - calling test(.)")
        trainer.test(ckpt_path='best', datamodule=data_module)
        
        print("test completed")
        print("done.")
    #< train
#< class