import torch
import torch.nn as nn
import timm
import os

class EfficientNetWrapper(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetWrapper, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

def convert_to_onnx(saved_model_path, output_onnx_path):
    # Ensure the device is CPU for ONNX conversion
    device = torch.device('cpu')
    
    # Load the saved model
    model = EfficientNetWrapper()
    model.load_state_dict(torch.load(saved_model_path, map_location=device)) # load the trained model checkpoint to restore weights
    model.to(device) # move to gpu
    model.eval() # perform the infernece

    # Create dummy input data to run a forward pass and conversion
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Convert and save as ONNX
    torch.onnx.export(model, dummy_input, output_onnx_path, verbose=True,)
    print(f"Model converted and saved to {output_onnx_path}")

if __name__ == "__main__":
    saved_model_path = "./saved_models/efficientnet_epoch_10.pth"
    if not os.path.exists(saved_model_path):
        print("Specified path does not exist.")
        exit()

    output_onnx_path = "./onnx_models/efficientnet_epoch_10_training.onnx"

    convert_to_onnx(saved_model_path, output_onnx_path)
