import torch
import numpy as np
import os
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.364)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, negative_slope=0.364)
        return out

class ChannelSpecificDecoder(nn.Module):
    def __init__(self, in_channels):
        super(ChannelSpecificDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.364, inplace=True),
            nn.Dropout2d(0.102),
            
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.364, inplace=True),
            nn.Dropout2d(0.102),
            
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.364, inplace=True),
            nn.Dropout2d(0.102),
            
            nn.ConvTranspose2d(in_channels // 8, 1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.364, inplace=True),
        )

    def forward(self, x):
        return self.decoder(x)

class EncoderDecoderCNN(nn.Module):
    def __init__(self):
        super(EncoderDecoderCNN, self).__init__()
        
        # Encoder layers with residual blocks
        self.encoder = nn.Sequential(
            ResidualBlock(1, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ResidualBlock(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ResidualBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layer for additional inputs (mach_norm and aoa_norm)
        self.fc_additional = nn.Linear(128 * 9 * 9 + 2, 128 * 9 * 9)
        
        # Individual decoders for each channel
        self.decoder1 = ChannelSpecificDecoder(128)
        self.decoder2 = ChannelSpecificDecoder(128)
        self.decoder3 = ChannelSpecificDecoder(128)
        self.decoder4 = ChannelSpecificDecoder(128)
        self.decoder5 = ChannelSpecificDecoder(128)

    def forward(self, sdf, mach_norm, aoa_norm):
        # Forward pass through encoder
        encoded = self.encoder(sdf)
        
        # Flatten or reshape encoded output to 1D vector
        flattened = torch.flatten(encoded, start_dim=1)
        
        # Concatenate mach_norm and aoa_norm to flattened output
        additional_inputs = torch.cat((mach_norm.unsqueeze(1), aoa_norm.unsqueeze(1)), dim=1)
        flattened_with_inputs = torch.cat((flattened, additional_inputs), dim=1)
        
        # Fully connected layer to match size before encoder
        flattened_out = self.fc_additional(flattened_with_inputs)
        
        # Reshape for decoder input
        reshaped = flattened_out.view(-1, 128, 9, 9)
        
        # Forward pass through individual decoders
        decoded1 = self.decoder1(reshaped)
        decoded2 = self.decoder2(reshaped)
        decoded3 = self.decoder3(reshaped)
        decoded4 = self.decoder4(reshaped)
        decoded5 = self.decoder5(reshaped)
        
        # Concatenate along the channel dimension
        decoded_concat = torch.cat((decoded1, decoded2, decoded3, decoded4, decoded5), dim=1)
        
        return decoded_concat

def fetch_data_for_inference(airfoil, mach, aoa, mach_norm, aoa_norm, sdf_folder_path):
    sdf = np.load(os.path.join(sdf_folder_path, f'{int(airfoil)}.npy'))
    
    sdf_tensor = torch.tensor(sdf, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    mach_norm_tensor = torch.tensor(mach_norm, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    aoa_norm_tensor = torch.tensor(aoa_norm, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    return sdf_tensor, mach_norm_tensor, aoa_norm_tensor

def inference(model, airfoil, mach, aoa, sdf_folder_path):
    # Timing measurement
    start_total = time.time()
    
    # Set device
    device = torch.device('cuda')
    model.to(device)
    
    # Load the scalers
    scaler_mach = MinMaxScaler()
    scaler_aoa = MinMaxScaler()
    scaler_mach.data_max_ = np.array([0.6])
    scaler_mach.data_min_ = np.array([0.4])
    scaler_mach.scale_ = np.array([1. / (scaler_mach.data_max_ - scaler_mach.data_min_)])
    scaler_mach.min_ = np.array([-2.])
    scaler_aoa.data_max_ = np.array([5.])
    scaler_aoa.data_min_ = np.array([0.])
    scaler_aoa.scale_ = np.array([1. / (scaler_aoa.data_max_ - scaler_aoa.data_min_)])
    scaler_aoa.min_ = np.array([0.])

    # Normalize Mach and AoA
    mach_norm = scaler_mach.transform(np.array([[mach]])).flatten()[0]
    aoa_norm = scaler_aoa.transform(np.array([[aoa]])).flatten()[0]

    # Fetch the input data
    sdf_tensor, mach_norm_tensor, aoa_norm_tensor = fetch_data_for_inference(airfoil, mach, aoa, mach_norm, aoa_norm, sdf_folder_path)
    
    # Move tensors to the device
    sdf_tensor = sdf_tensor.to(device)
    mach_norm_tensor = mach_norm_tensor.to(device)
    aoa_norm_tensor = aoa_norm_tensor.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        output = model(sdf_tensor, mach_norm_tensor, aoa_norm_tensor)
    
    # Convert output to numpy array and reshape
    output_np = output.cpu().numpy().squeeze()  # Remove the batch dimension
    output_np = np.transpose(output_np, (1, 2, 0))  # Change shape to (150, 150, 5)
    
    # Total execution time
    end_total = time.time()
    elapsed_total = end_total - start_total
    print(f'Time taken for Inference: {elapsed_total:.2f} seconds')
    
    return output_np

def model_init():
    
    start_total = time.time()
    
    # Path to the saved model
    #model_path = "E:/SU/Dis/DNN/model/new_model_10.pt" # Path to the file downloaded from google drive
    model_path = "../model/new_model_10.pt"
    
    # Load the saved model
    model = torch.load(model_path)
    model.eval()
    
    # Total execution time
    end_total = time.time()
    elapsed_total = end_total - start_total
    print(f'Time taken to initialise model: {elapsed_total:.2f} seconds')
    
    return model

def main_inference(airfoil, mach, aoa, model, sdf_folder_path = 'E:/SU/Dis/CFD/sdf_images', output_dir = 'E:/SU/Dis/DNN/Visualiser/inferred_data'):
    
    # Run the inference
    output = inference(model, airfoil, mach, aoa, sdf_folder_path)
    print(output.shape)
    
    # Save the output as a NumPy array
    output_file = os.path.join(output_dir, f'{airfoil}_{mach}_{aoa}.npy')
    np.save(output_file, output)
    print("Inference output saved")

#Example Implementation in console
# Define appropriate paths for folder with sdf arrays and location to store output in the main_inference function. "def main_inference(airfoil, mach, aoa, model, sdf_folder_path = 'E:/SU/Dis/CFD/sdf_images', output_dir = 'E:/SU/Dis/DNN/Visualiser/inferred_data'):"

#from DNN_UI import ResidualBlock, ChannelSpecificDecoder, EncoderDecoderCNN, model_init, main_inference
#model = model_init()
#main_inference(58,0.6,2,model)

#make sure to use sdf_generator to generate sdf before using DNN_UI for inference