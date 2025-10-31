import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from fusion_analysis import FusionAnalyzer
import os
import traceback

class FusionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo device detector")
        self.root.geometry("900x700")
        
        self.analyzer = FusionAnalyzer()
        self.setup_ui()
        
    def setup_ui(self):
        notebook = ttk.Notebook(self.root)

        training_frame = ttk.Frame(notebook)
        self.setup_training_tab(training_frame)

        detection_frame = ttk.Frame(notebook)
        self.setup_detection_tab(detection_frame)

        about_frame = ttk.Frame(notebook)
        self.setup_about_tab(about_frame)
        
        notebook.add(training_frame, text="Training")
        notebook.add(detection_frame, text="Device detection")
        notebook.add(about_frame, text="About")
        notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
    def setup_training_tab(self, parent):
        ttk.Label(parent, text="Select folder with training data").pack(pady=5)
        folder_frame = ttk.Frame(parent)
        folder_frame.pack(pady=5, fill='x')
        self.train_folder_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.train_folder_var, width=70).pack(side='left', padx=5, fill='x', expand=True)
        ttk.Button(folder_frame, text="Browse", command=self.browse_train_folder).pack(side='left')

        ttk.Button(parent, text="Start training", command=self.start_training).pack(pady=10)

        self.train_progress = ttk.Progressbar(parent, mode='indeterminate')
        self.train_progress.pack(pady=5, fill='x')

        log_frame = ttk.Frame(parent)
        log_frame.pack(pady=5, fill='both', expand=True)
        
        ttk.Label(log_frame, text="Training logs:").pack(anchor='w')
        self.train_log = tk.Text(log_frame, height=15, width=80)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.train_log.yview)
        self.train_log.configure(yscrollcommand=scrollbar.set)
        self.train_log.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
    def setup_detection_tab(self, parent):
        ttk.Label(parent, text="Select image for detection:").pack(pady=5)
        file_frame = ttk.Frame(parent)
        file_frame.pack(pady=5, fill='x')
        self.test_image_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.test_image_var, width=70).pack(side='left', padx=5, fill='x', expand=True)
        ttk.Button(file_frame, text="Browse", command=self.browse_test_image).pack(side='left')

        ttk.Button(parent, text="Analyze image", command=self.analyze_image).pack(pady=10)

        ttk.Label(parent, text="Analysis results:").pack(pady=5)
        results_notebook = ttk.Notebook(parent)
        results_notebook.pack(pady=5, fill='both', expand=True)
        

        jpeg_frame = ttk.Frame(results_notebook)
        self.jpeg_results = self.create_scrolled_text(jpeg_frame)
        results_notebook.add(jpeg_frame, text="JPEG analysis")
        



        prnu_frame = ttk.Frame(results_notebook)
        self.prnu_results = self.create_scrolled_text(prnu_frame)
        results_notebook.add(prnu_frame, text="PRNU analysis")
        cnn_frame = ttk.Frame(results_notebook)
        self.cnn_results = self.create_scrolled_text(cnn_frame)
        results_notebook.add(cnn_frame, text="CNN analysis")
        

        combined_frame = ttk.Frame(results_notebook)
        self.combined_results = self.create_scrolled_text(combined_frame)
        results_notebook.add(combined_frame, text="Combined results")
        
    def setup_about_tab(self, parent):
        about_text = """This program analyzes digital images using 3 methods.
The methods used are:

1. JPEG compression fingerprints
    // analyzes JPEG quantization matrices and DCT block artifacts
    // Creates signatures based on encoder patterns

2. PRNU (photo response non uniformity) fingerprinting
    // extracts unique sensor noise patterns
    // Compares noise residuals with stored references

3. Deep noise residual analysis (CNN-based)
   // uses convolutional neural network on image noise patterns

USAGE:

Train the models: 
1. Select a folder containing images with EXIF metadata (to group the images by device models and put model trained from the each device according to this device)
2. Wait till program trains the models

Detect device of the photo:
1. Detect the device of the selected photo in detection tab

* For best results, use high-quality JPEG images for training.
        """
        
        about_text_widget = tk.Text(parent, wrap=tk.WORD, padx=10, pady=10)
        about_text_widget.insert(1.0, about_text)
        about_text_widget.config(state=tk.DISABLED)
        about_text_widget.pack(fill='both', expand=True)
        
    def create_scrolled_text(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True)
        
        text_widget = tk.Text(frame, height=8)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        return text_widget
        
    def browse_train_folder(self):
        folder = filedialog.askdirectory(title="Select folder with photos for training")
        if folder:
            self.train_folder_var.set(folder)
            self.log_message(f"Selected training folder: {folder}")
            
    def browse_test_image(self):
        file_path = filedialog.askopenfilename(
            title="Select image for analysis",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tiff *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.test_image_var.set(file_path)
            self.log_message(f"Selected test image: {file_path}")
            
    def start_training(self):
        folder = self.train_folder_var.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Error", "Please select a valid training folder")
            return
            
        # Check if folder contains images
        image_files = [f for f in os.listdir(folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp'))]
        
        if not image_files:
            messagebox.showerror("Error", "No image files found in the selected folder")
            return
            
        # Run training in separate thread
        self.train_progress.start()
        self.log_message("Starting training...")
        self.log_message(f"Found {len(image_files)} image files")
        
        def train_thread():
            try:
                self.analyzer.train_models(folder, self.log_message)
                self.root.after(0, self.training_complete)
            except Exception as e:
                error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
                self.root.after(0, lambda msg=error_msg: self.training_error(msg))
                
        threading.Thread(target=train_thread, daemon=True).start()
        
    def training_complete(self):
        self.train_progress.stop()
        self.log_message("Training completed successfully!")
        messagebox.showinfo("Success", "Training completed successfully!")
        
    def training_error(self, error_msg):
        self.train_progress.stop()
        self.log_message(f"Training failed with error:")
        self.log_message(error_msg)
        messagebox.showerror("Error", f"Training failed. Check logs for details.")
        
    def analyze_image(self):
        image_path = self.test_image_var.get()
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please select a valid image file")
            return
            
        # Clear previous results
        for widget in [self.jpeg_results, self.prnu_results, self.cnn_results, self.combined_results]:
            widget.delete(1.0, tk.END)
            
        self.log_message(f"Starting analysis of: {os.path.basename(image_path)}")
        
        def analyze_thread():
            try:
                results = self.analyzer.analyze_image(image_path)
                self.root.after(0, lambda res=results, path=image_path: self.display_results(res, path))
            except Exception as e:
                error_msg = f"Analysis error: {str(e)}\n{traceback.format_exc()}"
                self.root.after(0, lambda msg=error_msg: self.analysis_error(msg))
                
        threading.Thread(target=analyze_thread, daemon=True).start()
        
    def display_results(self, results, image_path):
        # Display JPEG results
        self.jpeg_results.delete(1.0, tk.END)
        self.jpeg_results.insert(tk.END, f"JPEG Compression analysis results:\n")
        self.jpeg_results.insert(tk.END, "=" * 40 + "\n")
        if 'jpeg' in results:
            sorted_results = sorted(results['jpeg'].items(), key=lambda x: x[1], reverse=True)
            for device, score in sorted_results:
                self.jpeg_results.insert(tk.END, f"{device}: {score:.4f} ({score*100:.2f}%)\n")
        else:
            self.jpeg_results.insert(tk.END, "No JPEG analysis results available\n")
                
        # Display PRNU results
        self.prnu_results.delete(1.0, tk.END)
        self.prnu_results.insert(tk.END, f"PRNU sensor analysis results:\n")
        self.prnu_results.insert(tk.END, "=" * 40 + "\n")
        if 'prnu' in results:
            sorted_results = sorted(results['prnu'].items(), key=lambda x: x[1], reverse=True)
            for device, score in sorted_results:
                self.prnu_results.insert(tk.END, f"{device}: {score:.4f} ({score*100:.2f}%)\n")
        else:
            self.prnu_results.insert(tk.END, "No PRNU analysis results available\n")
                
        # Display CNN results
        self.cnn_results.delete(1.0, tk.END)
        self.cnn_results.insert(tk.END, f"CNN noise analysis results:\n")
        self.cnn_results.insert(tk.END, "=" * 40 + "\n")
        if 'cnn' in results:
            sorted_results = sorted(results['cnn'].items(), key=lambda x: x[1], reverse=True)
            for device, score in sorted_results:
                self.cnn_results.insert(tk.END, f"{device}: {score:.4f} ({score*100:.2f}%)\n")
        else:
            self.cnn_results.insert(tk.END, "No CNN analysis results available\n")
                
        # Display combined results
        self.combined_results.delete(1.0, tk.END)
        self.combined_results.insert(tk.END, f"Combined analysis results:\n")
        self.combined_results.insert(tk.END, "=" * 40 + "\n")
        if 'combined' in results:
            sorted_results = sorted(results['combined'].items(), key=lambda x: x[1], reverse=True)
            for i, (device, score) in enumerate(sorted_results):
                if i == 0:
                    self.combined_results.insert(tk.END, f"🏆 {device}: {score:.4f} ({score*100:.2f}%)\n")
                else:
                    self.combined_results.insert(tk.END, f"{device}: {score:.4f} ({score*100:.2f}%)\n")
        else:
            self.combined_results.insert(tk.END, "No combined analysis results available\n")
                
        self.log_message("Analysis completed successfully!")
        
    def analysis_error(self, error_msg):
        self.log_message(f"Analysis failed with error:")
        self.log_message(error_msg)
        messagebox.showerror("Error", f"Analysis failed. Check logs for details.")
        
    def log_message(self, message):
        def update_log():
            self.train_log.insert(tk.END, f"{message}\n")
            self.train_log.see(tk.END)
            self.root.update_idletasks()
            
        self.root.after(0, update_log)