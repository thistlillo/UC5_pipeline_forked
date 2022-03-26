
import pyeddl.eddl as eddl


def get_model(name: str):
    if name.startswith("resnet"):
        return get_resnet(name)
    else:
        assert False, f"unknown model {name}"
#<


def get_resnet(name):
    if name == "resnet18":
        base_cnn = eddl.download_resnet18(top=True)
