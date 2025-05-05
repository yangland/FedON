from models.convnet import ConvNet
from models.mnist_6l import Mnist_6L
from models.cifar10_18l import Cifar10_18L 
from models.ddpm import ContextUnet    
    

def get_model(model):
    # Dictionary mapping model names to their corresponding classes
    model_dict = {    
        "ConvNet": ConvNet,
        "Mnist_6L": Mnist_6L,
        "Cifar10_18L": Cifar10_18L,
        "ContextUnet": ContextUnet
    }

    # Check if the model name exists in the dictionary
    if model in model_dict:
        return model_dict[model]  # Return the model class as a tuple
    else:
        raise KeyError(f"Model '{model}' not found. Available models are: {', '.join(model_dict.keys())}")

