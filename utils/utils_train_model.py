from models import *

def get_model_list(device, num_node_features, hidden_channels):
    model_alex_only  = AlexNet_Only().to(device)
    model_VGG16_only = VGG16_Only().to(device)
    model_Restnet18_only = ResNet18_Only().to(device)
    model_EfficientNet_Only = EfficientNet_Only().to(device)
    model_GoogleNet_Only = GoogleNet_Only().to(device)
    
    model_Net_Alex = Net_Alex(num_node_features, hidden_channels).to(device)
    model_Net_Restnet = Net_ResNet18(num_node_features, hidden_channels).to(device)
    model_Net_EfficientNet = Net_EfficientNet(num_node_features, hidden_channels).to(device)
    model_Net_VGG = Net_VGG(num_node_features, hidden_channels).to(device)
    model_Net_Google = Net_GoogleNet(num_node_features, hidden_channels).to(device)
    
    model_dict = {
        # "model_alex_only":      model_alex_only,
        # "model_Restnet18_only": model_Restnet18_only,
        "model_EfficientNet_Only": model_EfficientNet_Only
        # "model_VGG16_only":     model_VGG16_only,
        # "model_GoogleNet_Only": model_GoogleNet_Only,
        
        # "model_Net_Alex":       model_Net_Alex,
        # "model_Net_Restnet":    model_Net_Restnet,
        # "model_Net_EfficientNet": model_Net_EfficientNet
        # "model_Net_VGG":        model_Net_VGG,
        # "model_Net_Google":     model_Net_Google
        
    }
    return model_dict