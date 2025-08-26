import omegaconf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from src import lit_module as lit

def train(config: omegaconf.DictConfig, checkpoint: str|None = None):
    lit_module = lit.LitModule(config)

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        accelerator=config.train.accelerator,
        callbacks=[ModelCheckpoint(save_last=True, save_top_k=5, monitor="train_loss")],
    )
    trainer.fit(lit_module, ckpt_path=checkpoint)