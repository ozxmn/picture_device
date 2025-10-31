import os
import json
import numpy as np
from PIL import Image, ExifTags
from database import Database
from jpeg_analyzer import JPEGAnalyzer
from prnu_analyzer import PRNUAnalyzer
from cnn_model import CNNModelWrapper
from utils import extract_exif_data, preprocess_image, get_supported_formats, is_image_file

class FusionAnalyzer:
    def __init__(self):
        self.db = Database()
        self.jpeg_analyzer = JPEGAnalyzer()
        self.prnu_analyzer = PRNUAnalyzer()
        self.cnn_model = CNNModelWrapper()
        self.is_trained = False
        self.load_models()
        
    def load_models(self):
        try:
            jpeg_loaded = self.jpeg_analyzer.load_model()
            prnu_loaded = self.prnu_analyzer.load_model()
            cnn_loaded = self.cnn_model.load_model()
            if jpeg_loaded or prnu_loaded or cnn_loaded:
                self.is_trained = True
                print(f"Loaded models: JPEG={jpeg_loaded}, PRNU={prnu_loaded}, CNN={cnn_loaded}")
                self.print_model_sizes()
            else:
                print("No saved models found, need to train first")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def print_model_sizes(self):
        model_dir = "models"
        total_size = 0
        if not os.path.exists(model_dir):
            return
        print("\n===== Model info=====")
        for filename in os.listdir(model_dir):
            filepath = os.path.join(model_dir, filename)
            size = os.path.getsize(filepath) / (1024 * 1024) 
            total_size += size
            print(f"{filename}: {size:.2f} MB")
        print(f"Total model size: {total_size:.2f} MB")
        print("================================\n")
        
    def train_models(self, training_folder, log_callback=None):
        return self.train_models_safe(training_folder, log_callback)
        
    def train_models_safe(self, training_folder, log_callback=None):
        if log_callback:
            log_callback("Scanning training folder for image files...")
            log_callback("EXIF extraction active...")
        device_images = {}
        processed_count = 0
        supported_count = 0
        unsupported_files = []
        device_detection_stats = {}
        for filename in os.listdir(training_folder):
            filepath = os.path.join(training_folder, filename)
            if is_image_file(filename):
                supported_count += 1
                processed_count += 1
                try:
                    exif_data = extract_exif_data(filepath)
                    device_model = exif_data.get('Detected_Device_Model', 'Unknown_Device')
                    detection_sources = exif_data.get('Device_Detection_Sources', {})
                    source_key = "_".join(detection_sources.keys())
                    device_detection_stats[source_key] = device_detection_stats.get(source_key, 0) + 1
                    if device_model not in device_images:
                        device_images[device_model] = []
                    device_images[device_model].append(filepath)
                    if log_callback and processed_count % 10 == 0:
                        log_callback(f"Processed {processed_count} images...")
                        log_callback(f"Current device: {device_model}")
                except Exception as e:
                    if log_callback:
                        log_callback(f"Error processing {filename}: {str(e)}")
            else:
                unsupported_files.append(filename)
        if log_callback:
            log_callback(f"Found {len(device_images)} device models")
            log_callback(f"Supported images: {supported_count}")
            log_callback(f"Unsupported files: {len(unsupported_files)}")
            log_callback(f"Total images processed: {processed_count}")
            log_callback("Device detection sources:")
            for source, count in device_detection_stats.items():
                log_callback(f"  {source}: {count} images")

            if unsupported_files:
                total_unsupported = len(unsupported_files)
                sample = unsupported_files[:5]
                log_callback(f"{total_unsupported} unsupported files found. First 5: {sample}")

        if not device_images:
            if log_callback:
                log_callback("No valid training images found. Training aborted.")
            return False
        methods = [
            ('JPEG', self.jpeg_analyzer.train, self.jpeg_analyzer.save_model),
            ('PRNU', self.prnu_analyzer.train, self.prnu_analyzer.save_model), 
            ('CNN', self.cnn_model.train, self.cnn_model.save_model)
        ]
        successful_methods = 0
        for method_name, train_func, save_func in methods:
            try:
                if log_callback:
                    log_callback(f"Training {method_name} analyzer...")
                train_func(device_images, log_callback)
                save_func()
                successful_methods += 1
                if log_callback:
                    log_callback(f"{method_name} training completed successfully")
            except Exception as e:
                if log_callback:
                    log_callback(f"{method_name} training failed: {str(e)}")
        self.is_trained = successful_methods > 0
        self.save_training_summary(device_images, successful_methods, unsupported_files, device_detection_stats, log_callback)
        self.print_model_sizes()
        if log_callback:
            if self.is_trained:
                log_callback(f"Training completed with {successful_methods}/3 methods successful")
                log_callback("All models saved  to 'models' directory")
            else:
                log_callback("Training failed for all methods")
        return self.is_trained
            
    def save_training_summary(self, device_images, successful_methods, unsupported_files, detection_stats, log_callback):
        summary = {
            'devices_trained': list(device_images.keys()),
            'samples_per_device': {k: len(v) for k, v in device_images.items()},
            'total_samples': sum(len(v) for v in device_images.values()),
            'successful_methods': successful_methods,
            'unsupported_files': unsupported_files,
            'detection_statistics': detection_stats,
            'supported_formats_count': len(get_supported_formats()),
            'timestamp': np.datetime64('now').astype(str),
            'storage_format': 'efficient_numpy_binary',
            'platform': 'cross_platform'
        }
        try:
            os.makedirs("models", exist_ok=True)
            with open('models/training_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            if log_callback:
                log_callback("Training summary saved to models/training_summary.json")
        except Exception as e:
            if log_callback:
                log_callback(f"Error saving training summary: {e}")
            
    def analyze_image(self, image_path):
        if not os.path.exists(image_path):
            return self.get_error_results(f"File not found: {image_path}")
        if not is_image_file(image_path):
            return self.get_error_results(f"Unsupported image format: {os.path.splitext(image_path)[1]}")
        if not self.is_trained:
            return self.get_demo_results()
        results = {}
        try:
            jpeg_results = self.jpeg_analyzer.analyze(image_path)
            results['jpeg'] = jpeg_results
            prnu_results = self.prnu_analyzer.analyze(image_path)
            results['prnu'] = prnu_results
            cnn_results = self.cnn_model.predict(image_path)
            results['cnn'] = cnn_results
            combined_results = self.combine_results(results)
            results['combined'] = combined_results
        except Exception as e:
            print(f"Analysis error for {image_path}: {e}")
            return self.get_error_results(f"Analysis failed: {str(e)}")
        return results
    
    def get_error_results(self, error_message):
        return {
            'jpeg': {error_message: 1.0},
            'prnu': {error_message: 1.0},
            'cnn': {error_message: 1.0},
            'combined': {error_message: 1.0}
        }
        
    def combine_results(self, results):
        all_devices = set()
        for method in ['jpeg', 'prnu', 'cnn']:
            if method in results:
                all_devices.update(results[method].keys())
        combined_results = {}
        for device in all_devices:
            scores = []
            weights = []
            if 'jpeg' in results and device in results['jpeg']:
                scores.append(results['jpeg'][device])
                weights.append(1.0)
            if 'prnu' in results and device in results['prnu']:
                scores.append(results['prnu'][device])
                weights.append(1.5)
            if 'cnn' in results and device in results['cnn']:
                scores.append(results['cnn'][device])
                weights.append(1.2)
            if scores:
                weighted_avg = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                combined_results[device] = weighted_avg
        total = sum(combined_results.values())
        if total > 0:
            combined_results = {k: v/total for k, v in combined_results.items()}
        return combined_results
        
    def get_demo_results(self):
        devices = []
        if hasattr(self.jpeg_analyzer, 'device_profiles') and self.jpeg_analyzer.device_profiles:
            devices.extend(self.jpeg_analyzer.device_profiles.keys())
        elif hasattr(self.prnu_analyzer, 'device_fingerprints') and self.prnu_analyzer.device_fingerprints:
            devices.extend(self.prnu_analyzer.device_fingerprints.keys())
        elif hasattr(self.cnn_model, 'idx_to_device') and self.cnn_model.idx_to_device:
            devices.extend(self.cnn_model.idx_to_device.values())
        if not devices:
            devices = ["iphone-demonstration", "samsung-demonstration", "canon-demonstration"]
        prob = 1.0 / len(devices)
        return {
            'jpeg': {device: prob for device in devices},
            'prnu': {device: prob for device in devices},
            'cnn': {device: prob for device in devices},
            'combined': {device: prob for device in devices}
        }
    
    def get_trained_devices(self):
        devices = set()
        if hasattr(self.jpeg_analyzer, 'device_profiles'):
            devices.update(self.jpeg_analyzer.device_profiles.keys())
        if hasattr(self.prnu_analyzer, 'device_fingerprints'):
            devices.update(self.prnu_analyzer.device_fingerprints.keys())
        if hasattr(self.cnn_model, 'idx_to_device'):
            devices.update(self.cnn_model.idx_to_device.values())
        return list(devices)
    
    def get_supported_formats_info(self):
        formats = get_supported_formats()
        categories = {
            'Standard': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
            'Web': ['.webp', '.avif', '.jxl'],
            'HEIC/HEIF': ['.heic', '.heif'],
            'RAW': [f for f in formats if f in [
                '.arw', '.cr2', '.cr3', '.nef', '.dng', '.raw', '.orf', '.pef', 
                '.srw', '.srf', '.sr2', '.mrw', '.dcr', '.kdc', '.erf', '.mef'
            ]],
            'Other': [f for f in formats if f not in [
                '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                '.webp', '.avif', '.jxl', '.heic', '.heif',
                '.arw', '.cr2', '.cr3', '.nef', '.dng', '.raw', '.orf', '.pef',
                '.srw', '.srf', '.sr2', '.mrw', '.dcr', '.kdc', '.erf', '.mef'
            ]]
        }
        return {
            'total_formats': len(formats),
            'categories': categories
        }
