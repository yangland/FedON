import inspect

from models.resnet import ResNet18_Light


def get_model(model_name):
    model_dict = {
        "ResNet18_Light": ResNet18_Light,
    }

    if model_name not in model_dict:
        raise KeyError(f"Model '{model_name}' not found. Available: {', '.join(model_dict.keys())}")

    model_class = model_dict[model_name]

    def model_wrapper(**kwargs):
        sig = inspect.signature(model_class.__init__)
        valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return model_class(**valid_args)

    return model_wrapper
