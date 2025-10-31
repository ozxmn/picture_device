import json
import os
import numpy as np

class Database:
    def __init__(self, db_path="fusion_db.json"):
        self.db_path = db_path
        self.data = self.load_data()
        
    def load_data(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "jpeg_profiles": {},
            "prnu_fingerprints": {},
            "cnn_model": {},
            "device_mapping": {}
        }
        
    def save_data(self):
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
            
    def save_jpeg_profile(self, device_model, profile):
        if "jpeg_profiles" not in self.data:
            self.data["jpeg_profiles"] = {}
        self.data["jpeg_profiles"][device_model] = profile.tolist() if isinstance(profile, np.ndarray) else profile
        self.save_data()
        
    def get_jpeg_profile(self, device_model):
        profile = self.data.get("jpeg_profiles", {}).get(device_model)
        if profile:
            return np.array(profile)
        return None
        
    def save_prnu_fingerprint(self, device_model, fingerprint):
        if "prnu_fingerprints" not in self.data:
            self.data["prnu_fingerprints"] = {}
        
        self.data["prnu_fingerprints"][device_model] = fingerprint.tolist() if isinstance(fingerprint, np.ndarray) else fingerprint
        self.save_data()
        
    def get_prnu_fingerprint(self, device_model):
        
        fingerprint = self.data.get("prnu_fingerprints", {}).get(device_model)
        if fingerprint:
            return np.array(fingerprint)
        return None