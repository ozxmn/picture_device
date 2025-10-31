import os
import json
import numpy as np
from scipy import ndimage
from scipy.fftpack import fft2, ifft2
from PIL import Image
from utils import load_image, wavelet_denoise

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


class PRNUAnalyzer:
    def __init__(self):
        self.device_fingerprints = {}
        self.fingerprint_size = (512, 512)

    def train(self, device_images, log_callback=None):
        self.device_fingerprints = {}

        for device_model, image_paths in device_images.items():
            if log_callback:
                log_callback(f"Training PRNU analyzer for {device_model}")

            fingerprints = []
            successful_extractions = 0

            for image_path in image_paths[:30]:
                try:
                    fp = self.extract_prnu_fingerprint(image_path)
                    if fp is not None and not np.allclose(fp, 0):
                        fingerprints.append(fp)
                        successful_extractions += 1
                except Exception as e:
                    if log_callback:
                        log_callback(f"Error processing {image_path}: {e}")

            if len(fingerprints) >= 3:
                standardized = self.standardize_fingerprints(fingerprints)
                arr = np.stack(standardized, axis=0)
                avg_fp = np.median(arr, axis=0)
                avg_fp = avg_fp - np.mean(avg_fp)
                self.device_fingerprints[device_model] = avg_fp.astype(np.float32)
                if log_callback:
                    log_callback(f"Successfully extracted {successful_extractions} fingerprints for {device_model}")
            else:
                if log_callback:
                    log_callback(f"Insufficient valid fingerprints for {device_model} ({len(fingerprints)} found)")

        if log_callback:
            log_callback("PRNU analyzer training completed")

    def standardize_fingerprints(self, fingerprints):
        target_h, target_w = self.fingerprint_size
        standardized = []
        for fp in fingerprints:
            fp_resized = self._resize_to_target(fp, (target_w, target_h))
            fp_resized = fp_resized.astype(np.float32)
            fp_resized = fp_resized - np.mean(fp_resized)
            standardized.append(fp_resized)
        return standardized

    def extract_prnu_fingerprint(self, image_path):
        try:
            img = load_image(image_path)
            if img is None:
                return None

            if img.ndim == 3:
                img = img[:, :, 1] if img.shape[2] >= 2 else np.mean(img, axis=2)

            img = img.astype(np.float32)

            target_h, target_w = self.fingerprint_size
            img_resized = self._resize_to_target(img, (target_w, target_h)).astype(np.float32)

            residual = None
            try:
                den = wavelet_denoise(img_resized)
                residual = img_resized - den
            except Exception:
                den = ndimage.gaussian_filter(img_resized, sigma=1.0)
                residual = img_resized - den

            residual = residual - np.mean(residual)

            filtered = self._wiener_and_highpass(residual)
            filtered = filtered - np.mean(filtered)
            return filtered.astype(np.float32)

        except Exception as e:
            print(f"Error extracting PRNU from {image_path}: {e}")
            return None

    def _resize_to_target(self, img, size_wh):
        w, h = size_wh
        try:
            if _HAS_CV2:
                resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
                return resized
        except Exception:
            pass

        # fallback to scipy.ndimage.zoom while preserving aspect with cropping/padding
        src_h, src_w = img.shape
        zoom_y = h / src_h
        zoom_x = w / src_w
        resized = ndimage.zoom(img, (zoom_y, zoom_x))
        if resized.shape != (h, w):
            resized = resized[:h, :w]
            if resized.shape != (h, w):
                pad_h = h - resized.shape[0]
                pad_w = w - resized.shape[1]
                resized = np.pad(resized, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode='edge')
        return resized

    def _wiener_and_highpass(self, residual):
        M, N = residual.shape
        eps = 1e-12

        F = fft2(residual)
        P = (np.abs(F) ** 2) / (M * N)

        noise_floor = np.median(P)
        noise_power = max(noise_floor, eps)

        signal_est = np.maximum(P - noise_power, 0.0)

        H = signal_est / (signal_est + noise_power + eps)
        F_filtered = F * H
        filtered = np.real(ifft2(F_filtered))

        # design a smooth high-pass mask relative to Nyquist
        u = np.fft.fftfreq(M)
        v = np.fft.fftfreq(N)
        U, V = np.meshgrid(u, v, indexing='ij')
        D = np.sqrt(U ** 2 + V ** 2)

        cutoff = 0.02
        transition = 0.01
        hp = 1.0 - 1.0 / (1.0 + np.exp((D - cutoff) / (transition + eps)))

        F2 = fft2(filtered) * hp
        filtered_hp = np.real(ifft2(F2))
        return filtered_hp

    def analyze(self, image_path):
        if not self.device_fingerprints:
            return {"No trained models": 1.0}

        test_fp = self.extract_prnu_fingerprint(image_path)
        if test_fp is None or np.allclose(test_fp, 0):
            return {"Error": 1.0}

        test_norm = self._normalize_for_comparison(test_fp)

        correlations = {}
        for device, ref_fp in self.device_fingerprints.items():
            ref_norm = self._normalize_for_comparison(ref_fp)
            score = self.compute_enhanced_similarity(test_norm, ref_norm)
            correlations[device] = score

        if not correlations:
            devices = list(self.device_fingerprints.keys())
            prob = 1.0 / len(devices) if devices else 1.0
            return {d: prob for d in devices}

        max_corr = max(correlations.values())
        scale = 10.0
        exp_corrs = {k: float(np.exp(scale * (v - max_corr))) for k, v in correlations.items()}
        total = sum(exp_corrs.values())
        results = {k: float(v / total) for k, v in exp_corrs.items()}
        return results

    def _normalize_for_comparison(self, fp):
        mu = np.mean(fp)
        sigma = np.std(fp)
        if sigma < 1e-8:
            return fp - mu
        return (fp - mu) / sigma

    def compute_enhanced_similarity(self, img1, img2):
        try:
            f1 = img1.flatten()
            f2 = img2.flatten()

            denom = (np.sqrt(np.sum(f1 ** 2) * np.sum(f2 ** 2)) + 1e-12)
            pearson = float(np.sum(f1 * f2) / denom)

            F1 = np.fft.fft2(img1)
            F2 = np.fft.fft2(img2)
            cross = F1 * np.conj(F2)
            denom_cross = np.abs(cross)
            denom_cross[denom_cross == 0] = 1e-12
            cross_norm = cross / denom_cross
            corr = np.fft.ifft2(cross_norm)
            corr_abs = np.abs(corr)
            peak = float(np.max(corr_abs))
            poc_score = (peak - 1.0) / (np.sqrt(img1.size) + 1e-12)
            poc_score = float(np.clip(poc_score, 0.0, 1.0))

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            var1 = np.var(img1)
            var2 = np.var(img2)
            cov12 = np.mean((img1 - mu1) * (img2 - mu2))
            denom_ssim = (mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2)
            if denom_ssim <= 0:
                ssim_like = 0.0
            else:
                ssim_like = ((2 * mu1 * mu2 + C1) * (2 * cov12 + C2)) / denom_ssim
                ssim_like = float(np.clip(ssim_like, 0.0, 1.0))

            combined = 0.7 * max(0.0, pearson) + 0.2 * poc_score + 0.1 * ssim_like
            return float(max(0.0, combined))

        except Exception as e:
            print(f"Enhanced similarity computation error: {e}")
            return 0.0

    def save_model(self, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        model_data = {}
        for device, fingerprint in self.device_fingerprints.items():
            model_data[device] = fingerprint.tolist()
        with open(os.path.join(model_dir, "prnu_fingerprints.json"), "w") as f:
            json.dump(model_data, f)

    def load_model(self, model_dir="models"):
        model_path = os.path.join(model_dir, "prnu_fingerprints.json")
        if os.path.exists(model_path):
            try:
                with open(model_path, "r") as f:
                    model_data = json.load(f)
                self.device_fingerprints = {}
                for device, fp_list in model_data.items():
                    self.device_fingerprints[device] = np.array(fp_list, dtype=np.float32)
                return True
            except Exception as e:
                print(f"Error loading PRNU model: {e}")
        return False
