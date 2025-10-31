import numpy as np
from PIL import Image
import os
import json
from utils import load_image, dct2, compare_histograms

class JPEGAnalyzer:
    def __init__(self):
        self.device_profiles = {}
        self.feature_dtype = np.float32
        
    def train(self, device_images, log_callback=None):
        self.device_profiles = {}
        
        for device_model, image_paths in device_images.items():
            if log_callback:
                log_callback(f"Training JPEG analyzer for {device_model}")
                
            features = []
            
            for image_path in image_paths[:30]: 
                try:
                    jpeg_features = self.extract_jpeg_features(image_path)
                    if jpeg_features is not None:
                        features.append(jpeg_features)
                except Exception as e:
                    if log_callback:
                        log_callback(f"Error processing {image_path}: {str(e)}")
        
            if features:
                standardized_features = self.ensure_consistent_features(features)
                features_array = np.array(standardized_features)
                avg_features = np.mean(features_array, axis=0)
                self.device_profiles[device_model] = avg_features.astype(self.feature_dtype)
                
                if log_callback:
                    log_callback(f"Trained {device_model} with {len(features)} samples")
            else:
                if log_callback:
                    log_callback(f"No features for {device_model}")
            
        if log_callback:
            log_callback("JPEG analyzer training completed")
            
    def ensure_consistent_features(self, features_list):
        if not features_list:
            return []
        
        max_length = max(len(features) for features in features_list)
        
        standardized_features = []
        for features in features_list:
            if len(features) < max_length:
                padded = np.pad(features, (0, max_length - len(features)), mode='constant')
                standardized_features.append(padded)
            else:
                standardized_features.append(features[:max_length])
        
        return standardized_features
            
    def extract_jpeg_features(self, image_path):
        try:
            img = load_image(image_path)
            if img is None:
                return None
                
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
                
            h, w = img.shape
            h = h - (h % 8)
            w = w - (w % 8)
            if h < 8 or w < 8:
                return None
            img = img[:h, :w]
            
            features = []
            
            dct_coeffs = self.compute_dct_statistics(img)
            features.extend(dct_coeffs)
            block_features = self.compute_block_artifacts(img)
            features.extend(block_features)
            
            return np.array(features, dtype=self.feature_dtype)
            
        except Exception as e:
            print(f"Error extracting JPEG features: {e}")
            return None
            
    def compute_dct_statistics(self, img):
        h, w = img.shape
        dct_blocks = []
        
        for i in range(0, h-7, 32):
            for j in range(0, w-7, 32):
                if i < h-7 and j < w-7:
                    block = img[i:i+8, j:j+8]
                    dct_block = dct2(block)
                    dct_blocks.append(dct_block.flatten())
                
        if not dct_blocks:
            return [0] * 12
            
        dct_blocks = np.array(dct_blocks)

        features = []
        for k in range(min(12, dct_blocks.shape[1])):
            coeffs = dct_blocks[:, k]
            if len(coeffs) > 0:
                features.extend([
                    np.mean(coeffs),
                    np.std(coeffs)
                ])
            else:
                features.extend([0, 0])
            
        return features[:12]
        
    def compute_block_artifacts(self, img):
        h, w = img.shape
        features = []
        
        horizontal_diffs = []
        for i in range(8, min(h, 256), 8):
            diff = np.mean(np.abs(img[i, :min(w, 256)] - img[i-1, :min(w, 256)]))
            horizontal_diffs.append(diff)
                
        vertical_diffs = []
        for j in range(8, min(w, 256), 8):
            diff = np.mean(np.abs(img[:min(h, 256), j] - img[:min(h, 256), j-1]))
            vertical_diffs.append(diff)
                
        if horizontal_diffs:
            features.extend([
                np.mean(horizontal_diffs),
                np.std(horizontal_diffs)
            ])
        else:
            features.extend([0, 0])
            
        if vertical_diffs:
            features.extend([
                np.mean(vertical_diffs),
                np.std(vertical_diffs)
            ])
        else:
            features.extend([0, 0])
            
        return features
        
    def analyze(self, image_path):
        if not self.device_profiles:
            return {"No trained models": 1.0}
            
        test_features = self.extract_jpeg_features(image_path)
        if test_features is None:
            return {"Error": 1.0}
            
        results = {}
        distances = {}
        
        for device, profile in self.device_profiles.items():
            min_len = min(len(test_features), len(profile))
            if min_len == 0:
                continue
                
            test_features_trunc = test_features[:min_len]
            profile_trunc = profile[:min_len]
            
            diff = test_features_trunc - profile_trunc
            std_profile = np.std(profile_trunc)
            if std_profile > 0:
                normalized_diff = diff / std_profile
                distance = np.sqrt(np.mean(normalized_diff**2))
            else:
                distance = np.linalg.norm(diff)
                
            similarity = 1.0 / (1.0 + distance)
            distances[device] = similarity
            
        #softmax
        if distances:
            max_sim = max(distances.values())
            exp_sims = {k: np.exp(3.0 * (v - max_sim)) for k, v in distances.items()}
            total = sum(exp_sims.values())
            results = {k: v/total for k, v in exp_sims.items()}
        else:
            devices = list(self.device_profiles.keys())
            prob = 1.0 / len(devices) if devices else 1.0
            results = {device: prob for device in devices}
            
        return results

    def save_model(self, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        
        if not self.device_profiles:
            return
            
        device_names = list(self.device_profiles.keys())
        profile_stack = np.stack([self.device_profiles[name] for name in device_names], axis=0)
        np.save(os.path.join(model_dir, "jpeg_profiles_2d.npy"), profile_stack)
        
        model_info = {
            'device_names': device_names,
            'feature_dtype': str(self.feature_dtype)
        }
        
        with open(os.path.join(model_dir, "jpeg_model_info.json"), 'w') as f:
            json.dump(model_info, f)
            
    def load_model(self, model_dir="models"):
        npy_path = os.path.join(model_dir, "jpeg_profiles_2d.npy")
        info_path = os.path.join(model_dir, "jpeg_model_info.json")
        
        if os.path.exists(npy_path) and os.path.exists(info_path):
            try:
                profile_stack = np.load(npy_path)
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                device_names = model_info['device_names']
                
                self.device_profiles = {}
                for i, device_name in enumerate(device_names):
                    self.device_profiles[device_name] = profile_stack[i]
                    
                return True
            except Exception as e:
                print(f"Error loading JPEG model: {e}")
        return False