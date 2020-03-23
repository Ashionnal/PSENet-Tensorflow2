
from core.nets.psenet import PSENet
from utils.infetence import eval_model
import os
from config import model_dir

psenet = PSENet()
psenet.load_weights(filepath=os.path.join(model_dir, 'model-90500'))
if __name__ == "__main__":
    eval_model(psenet)