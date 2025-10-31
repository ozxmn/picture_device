import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError
import scipy.fftpack as fftpack
from scipy import ndimage
import os
import rawpy
import imageio
import warnings
from typing import Optional, Union
import pillow_heif

pillow_heif.register_heif_opener()

def load_image(image_path: str) -> Optional[np.ndarray]:
    try:
        ext = os.path.splitext(image_path.lower())[1]
        if ext in ['.arw', '.cr2', '.nef', '.dng', '.raw', '.orf', '.pef', '.sr2', '.srw']:
            return load_raw_image(image_path)
        if ext in ['.heic', '.heif']:
            return load_heic_image(image_path)
        if ext in ['.webp', '.avif', '.jxl']:
            return load_special_format(image_path)
        return load_standard_image(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def load_raw_image(image_path: str) -> Optional[np.ndarray]:
    try:
        with rawpy.imread(image_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=8
            )
            return rgb
    except Exception as e:
        print(f"Error loading RAW image {image_path}: {e}")
        return load_standard_image(image_path)

def load_heic_image(image_path: str) -> Optional[np.ndarray]:
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
    except Exception as e:
        print(f"Error loading HEIC image {image_path}: {e}")
        return None

def load_special_format(image_path: str) -> Optional[np.ndarray]:
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
    except Exception as e:
        print(f"Error loading special format image {image_path}: {e}")
        return None

def load_standard_image(image_path: str) -> Optional[np.ndarray]:
    try:
        with Image.open(image_path) as img:
            if img.mode in ('1', 'L', 'P'):
                img = img.convert('RGB')
            elif img.mode == 'LA':
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode == 'CMYK':
                img = img.convert('RGB')
            return np.array(img)
    except UnidentifiedImageError:
        print(f"Cannot identify image file {image_path}")
        return None
    except Exception as e:
        print(f"Error loading standard image {image_path}: {e}")
        return None

def extract_exif_data_enhanced(image_path: str) -> dict:
    exif_data = {}
    try:
        with Image.open(image_path) as img:
            exif_data['Format'] = img.format
            exif_data['Size'] = img.size
            exif_data['Mode'] = img.mode
            exif_data['Width'] = img.width
            exif_data['Height'] = img.height
            exif = None
            try:
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                elif hasattr(img, 'getexif'):
                    exif = img.getexif()
            except:
                pass
            if exif:
                for tag, value in exif.items():
                    tag_name = Image.ExifTags.TAGS.get(tag, tag)
                    if isinstance(value, (str, int, float, bytes)):
                        try:
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode('utf-8', errors='ignore')
                                except:
                                    value = str(value)
                            exif_data[tag_name] = str(value)
                        except:
                            exif_data[tag_name] = "Unreadable"
            try:
                for key, value in img.info.items():
                    if key not in exif_data and key not in ['exif', 'icc_profile']:
                        try:
                            exif_data[f"Info_{key}"] = str(value)
                        except:
                            pass
            except:
                pass
    except Exception as e:
        exif_data['Error'] = f"PIL extraction failed: {str(e)}"
    exif_data = enhance_device_detection(exif_data, image_path)
    return exif_data

def enhance_device_detection(exif_data: dict, image_path: str) -> dict:
    device_sources = [
        ('Model', 'Model'),
        ('Camera Model Name', 'Camera_Model'),
        ('Make', 'Make'),
        ('Software', 'Software'),
        ('Processing Software', 'Processing_Software'),
        ('LensModel', 'Lens_Model'),
        ('LensMake', 'Lens_Make'),
        ('XPTitle', 'XP_Title'),
        ('XPComment', 'XP_Comment'),
        ('XPAuthor', 'XP_Author'),
        ('XPSubject', 'XP_Subject'),
        ('XPKeywords', 'XP_Keywords'),
    ]
    device_info = {}
    for exif_key, info_key in device_sources:
        if exif_key in exif_data and exif_data[exif_key] and exif_data[exif_key] not in ['', 'Unknown', 'undefined']:
            device_info[info_key] = exif_data[exif_key]
    final_model = "Unknown_Device"
    if 'Model' in device_info:
        final_model = device_info['Model']
    elif 'Camera_Model' in device_info:
        final_model = device_info['Camera_Model']
    elif 'Make' in device_info:
        make = device_info['Make']
        model_clues = []
        if 'Software' in device_info:
            software = device_info['Software']
            software_lower = software.lower()
            if 'iphone' in software_lower:
                model_clues.append('iPhone')
            elif 'samsung' in software_lower:
                model_clues.append('Galaxy')
            elif 'google' in software_lower:
                model_clues.append('Pixel')
            elif 'huawei' in software_lower:
                model_clues.append('Huawei')
            elif 'xiaomi' in software_lower:
                model_clues.append('Xiaomi')
            elif 'oneplus' in software_lower:
                model_clues.append('OnePlus')
        if 'Lens_Model' in device_info:
            lens = device_info['Lens_Model'].lower()
            if 'front' in lens or 'selfie' in lens:
                model_clues.append('Front_Camera')
            elif 'back' in lens or 'main' in lens or 'rear' in lens:
                model_clues.append('Back_Camera')
        for xp_key in ['XP_Title', 'XP_Comment', 'XP_Author', 'XP_Subject', 'XP_Keywords']:
            if xp_key in device_info:
                xp_value = device_info[xp_key].lower()
                if 'iphone' in xp_value:
                    model_clues.append('iPhone')
                elif 'samsung' in xp_value:
                    model_clues.append('Galaxy')
        if model_clues:
            final_model = f"{make}_{'_'.join(model_clues)}"
        else:
            final_model = f"{make}_Device"
    final_model = clean_device_name(final_model)
    exif_data['Detected_Device_Model'] = final_model
    exif_data['Device_Detection_Sources'] = device_info
    return exif_data

def clean_device_name(device_name: str) -> str:
    if not device_name or device_name == 'Unknown_Device':
        return 'Unknown_Device'
    replacements = {
        ' ': '_',
        '-': '_',
        '.': '',
        '/': '_',
        '\\': '_',
        ':': '',
        ';': '',
        ',': '',
        '__': '_',
        '___': '_'
    }
    cleaned = device_name.strip()
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    garbage_patterns = [
        'undefined', 'null', 'unknown', 'default', 
        'front_camera', 'back_camera', 'main_camera', 'rear_camera'
    ]
    for pattern in garbage_patterns:
        cleaned = cleaned.replace(pattern, '')
    cleaned = cleaned.strip('_')
    if not cleaned:
        return 'Unknown_Device'
    return cleaned

def extract_exif_data(image_path: str) -> dict:
    return extract_exif_data_enhanced(image_path)

def dct2(block: np.ndarray) -> np.ndarray:
    try:
        return fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
    except:
        return np.zeros_like(block)

def idct2(block: np.ndarray) -> np.ndarray:
    try:
        return fftpack.idct(fftpack.idct(block.T, norm='ortho').T, norm='ortho')
    except:
        return np.zeros_like(block)

def wavelet_denoise(img: np.ndarray, wavelet: str = 'db4', level: int = 1) -> np.ndarray:
    try:
        import pywt
        img_float = img.astype(np.float64)
        if np.max(img_float) > 1.0:
            img_float = img_float / 255.0
        coeffs = pywt.wavedec2(img_float, wavelet, level=level)
        coeffs_thresh = [coeffs[0]]
        for i in range(1, len(coeffs)):
            detail = list(coeffs[i])
            threshold = np.std(detail[0]) * np.sqrt(2 * np.log(img_float.size))
            detail_thresh = [np.sign(c) * np.maximum(np.abs(c) - threshold, 0) for c in detail]
            coeffs_thresh.append(tuple(detail_thresh))
        denoised = pywt.waverec2(coeffs_thresh, wavelet)
        if denoised.shape != img_float.shape:
            zoom_factors = (img_float.shape[0]/denoised.shape[0], img_float.shape[1]/denoised.shape[1])
            denoised = ndimage.zoom(denoised, zoom_factors)
        if np.max(img) > 1.0:
            denoised = denoised * 255.0
        return denoised.astype(img.dtype)
    except ImportError:
        warnings.warn("PyWavelets not available, using Gaussian filter fallback")
        return ndimage.gaussian_filter(img, sigma=1)
    except Exception as e:
        warnings.warn(f"Wavelet denoising failed: {e}, using Gaussian filter")
        return ndimage.gaussian_filter(img, sigma=1)

def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    eps = 1e-10
    return 0.5 * np.sum((hist1 - hist2)**2 / (hist1 + hist2 + eps))

def preprocess_image(img: np.ndarray, target_size: tuple = (256, 256)) -> np.ndarray:
    try:
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        if img.shape != target_size:
            zoom_factors = (target_size[0]/img.shape[0], target_size[1]/img.shape[1])
            img = ndimage.zoom(img, zoom_factors)
        return img
    except:
        return img

def get_supported_formats() -> list:
    base_formats = [
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
        '.webp', '.avif', '.jxl',
        '.heic', '.heif',
        '.arw', '.cr2', '.cr3', '.nef', '.nrw', '.dng', '.raw', 
        '.orf', '.pef', '.ptx', '.srw', '.srf', '.sr2',
        '.mrw', '.mdc', '.kdc', '.dcr', '.dcs', '.drf',
        '.k25', '.kc2', '.erf', '.mef', '.mos', '.fff',
        '.raf', '.rw2', '.rwl', '.srw', '.x3f',
        '.ico', '.icns', '.ppm', '.pgm', '.pbm', '.pnm',
        '.hdr', '.exr', '.jp2', '.j2k', '.jpf', '.jpx',
        '.tga', '.sgi', '.bw', '.rgb', '.rgba', '.pcd',
        '.pcx', '.cut', '.xbm', '.xpm'
    ]
    return base_formats

def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in get_supported_formats()

def get_image_info(image_path: str) -> dict:
    info = {}
    try:
        with Image.open(image_path) as img:
            info['format'] = img.format
            info['size'] = img.size
            info['mode'] = img.mode
            info['width'] = img.width
            info['height'] = img.height
            stat = os.stat(image_path)
            info['file_size'] = stat.st_size
            info['modified_time'] = stat.st_mtime
            try:
                if hasattr(img, 'info'):
                    info['image_info'] = dict(img.info)
            except:
                pass
    except Exception as e:
        info['error'] = str(e)
    return info

def convert_to_standard_format(image_path: str, output_path: str = None) -> str:
    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + '_converted.jpg'
    try:
        img_array = load_image(image_path)
        if img_array is not None:
            img = Image.fromarray(img_array)
            img.save(output_path, 'JPEG', quality=95)
            return output_path
    except Exception as e:
        print(f"Error converting image {image_path}: {e}")
    return image_path
