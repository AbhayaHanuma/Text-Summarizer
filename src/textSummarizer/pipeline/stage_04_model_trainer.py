from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.logging import logger
import os


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        if os.path.exists(os.path.join(model_trainer_config.root_dir,"pegasus-samsum-model")) and os.path.exists(os.path.join(model_trainer_config.root_dir,"tokenizer")):
            logger.info(f"Trained model already exists so skipping the training")
        else:
            model_trainer_config = ModelTrainer(config=model_trainer_config)
            model_trainer_config.train()