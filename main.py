from src.data.make_dataset import main as make_dataset
from src.models.train_model import main as train_model

from src.logging import logger


def main():
    make_dataset()
    train_model()

if __name__ == "__main__":
    logger.info("Main program started")
    main()
    logger.info("Main program excecuted successfully")