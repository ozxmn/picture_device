import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import json
from utils import load_image, wavelet_denoise

class NoiseDataset(Dataset):
    def __init__(self, device_images, transform=None):
        self.samples = []
        self.device_to_idx = {}
        self.idx_to_device = {}
        
        device_idx = 0
        for device, image_paths in device_images.items():
            self.device_to_idx[device] = device_idx
            self.idx_to_device[device_idx] = device
            device_idx += 1
            
            for img_path in image_paths[:50]:  
                self.samples.append((img_path, self.device_to_idx[device]))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, device_idx = self.samples[idx]
        
        try:
            img = load_image(img_path)
            if img is None:
                return torch.zeros(1, 64, 64), device_idx
                
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
                

            try:
                denoised = wavelet_denoise(img)
                noise_residual = img - denoised
            except:
                from scipy.ndimage import gaussian_filter
                denoised = gaussian_filter(img, sigma=1)
                noise_residual = img - denoised
            
            if np.std(noise_residual) > 0:
                noise_residual = (noise_residual - np.mean(noise_residual)) / np.std(noise_residual)
            else:
                noise_residual = np.zeros_like(img)
            
            h, w = noise_residual.shape
            if h < 64 or w < 64:
                pad_h = max(0, 64 - h)
                pad_w = max(0, 64 - w)
                noise_residual = np.pad(noise_residual, ((0, pad_h), (0, pad_w)), mode='constant')
            else:
                noise_residual = noise_residual[:64, :64]
            
            noise_tensor = torch.FloatTensor(noise_residual).unsqueeze(0)
            
            return noise_tensor, device_idx
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(1, 64, 64), device_idx

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNNModelWrapper:
    def __init__(self):
        self.model = None
        self.device_to_idx = {}
        self.idx_to_device = {}
        self.is_trained = False
        
    def train(self, device_images, log_callback=None):
        if not device_images:
            if log_callback:
                log_callback("No training data available for CNN")
            return
            
        # Create dataset
        dataset = NoiseDataset(device_images)
        self.device_to_idx = dataset.device_to_idx
        self.idx_to_device = dataset.idx_to_device
        
        num_classes = len(self.device_to_idx)
        
        if num_classes < 2:
            if log_callback:
                log_callback("Need at least 2 device classes for CNN training")
            return
            
        if log_callback:
            log_callback(f"Training CNN with {num_classes} device classes")
            log_callback(f"Using {len(dataset)} training samples")

        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        
        self.model = CNNModel(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        num_periods = 5
        
        for period in range(num_periods):
            if log_callback:
                log_callback(f"CNN period {period+1}/{num_periods}")
                
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
            accuracy = 100 * correct / total if total > 0 else 0
            if log_callback:
                log_callback(f"Loss: {running_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")
                
        self.is_trained = True
        
        if log_callback:
            log_callback("CNN training completed")
            
    def predict(self, image_path):
        """Predict device probabilities using CNN"""
        if not self.is_trained or self.model is None:
            return self.get_dummy_predictions()
            
        try:
            img = load_image(image_path)
            if img is None:
                return self.get_dummy_predictions()
                
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
                
            try:
                denoised = wavelet_denoise(img)
                noise_residual = img - denoised
            except:
                from scipy.ndimage import gaussian_filter
                denoised = gaussian_filter(img, sigma=1)
                noise_residual = img - denoised
            
            if np.std(noise_residual) > 0:
                noise_residual = (noise_residual - np.mean(noise_residual)) / np.std(noise_residual)
            else:
                noise_residual = np.zeros_like(img)
            
            h, w = noise_residual.shape
            if h < 64 or w < 64:
                noise_residual = np.pad(noise_residual, 
                                      ((0, max(0, 64-h)), (0, max(0, 64-w))), 
                                      mode='constant')
            else:
                noise_residual = noise_residual[:64, :64]
            
            input_tensor = torch.FloatTensor(noise_residual).unsqueeze(0).unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1).squeeze().numpy()
                
            results = {}
            for idx, prob in enumerate(probabilities):
                device_name = self.idx_to_device.get(idx, f"Device_{idx}")
                results[device_name] = float(prob)
                
            return results
            
        except Exception as e:
            print(f"CNN prediction error: {e}")
            return self.get_dummy_predictions()
            
    def get_dummy_predictions(self):
        if self.idx_to_device:
            devices = list(self.idx_to_device.values())
            prob = 1.0 / len(devices) if devices else 1.0
            return {device: prob for device in devices}
        else:
            return {"demo_device": 1.0}

    def save_model(self, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(model_dir, "cnn_model.pth"))
            
        model_info = {
            'device_to_idx': self.device_to_idx,
            'idx_to_device': self.idx_to_device
        }
        
        with open(os.path.join(model_dir, "cnn_model_info.json"), 'w') as f:
            json.dump(model_info, f)
            
    def load_model(self, model_dir="models"):
        model_path = os.path.join(model_dir, "cnn_model.pth")
        info_path = os.path.join(model_dir, "cnn_model_info.json")
        
        if os.path.exists(model_path) and os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                    self.device_to_idx = model_info['device_to_idx']
                    self.idx_to_device = {int(k): v for k, v in model_info['idx_to_device'].items()}
                
                num_classes = len(self.device_to_idx)
                if num_classes > 0:
                    self.model = CNNModel(num_classes)
                    self.model.load_state_dict(torch.load(model_path))
                    self.model.eval()
                    self.is_trained = True
                    return True
                    
            except Exception as e:
                print(f"Error loading CNN model: {e}")
                
        return False