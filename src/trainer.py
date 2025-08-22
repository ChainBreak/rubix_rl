import omegaconf
import lightning as L
from src import lit_module as lit

def train(config: omegaconf.DictConfig):
    lit_module = lit.LitModule(config)
    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        accelerator=config.train.accelerator,
        # logger=config.train.logger,
    )
    trainer.fit(lit_module)