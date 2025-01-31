from tools.trainer import Trainer
import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(3407)
    trainer = Trainer(f"config/pems03.json")
    # trainer = Trainer(f"config/pems04.json")
    trainer.fit()
    trainer.predict()
