from model import *

def get_model_list(device, num_node_features, hidden_channels):
    model_alex_only  = AlexNet_Only().to(device)
    model_VGG19_only = VGG19_Only().to(device)
    model_Restnet18_only = ResNet18_Only().to(device)
    model_Net_Alex = Net_Alex(num_node_features, hidden_channels).to(device)
    model_Net_VGG = Net_VGG(num_node_features, hidden_channels).to(device)
    model_Net_Restnet = Net_ResNet18(num_node_features, hidden_channels).to(device)
    model_dict = {
        "model_alex_only":      model_alex_only,
        "model_Net_Alex":       model_Net_Alex,
        "model_VGG19_only":     model_VGG19_only,
        "model_Net_VGG":        model_Net_VGG,
        "model_Restnet18_only": model_Restnet18_only,
        "model_Net_Restnet":    model_Net_Restnet
    }
    return model_dict