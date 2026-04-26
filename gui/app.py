import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
import threading

IMG_SIZE = 128


class ModernSkinAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mobile Skin Analyzer")
        self.root.geometry("600x800")
        self.root.configure(bg="#1a1a2e")
        
        self.model = None
        self.interpreter = None
        self.current_image_path = None
        self.processed_image = None
        
        self.load_model()
        self.create_ui()
        
    def load_model(self):
        try:
            if os.path.exists('model/skin_cancer_model.tflite'):
                self.interpreter = tf.lite.Interpreter(model_path='model/skin_cancer_model.tflite')
                self.interpreter.allocate_tensors()
                print("Loaded TFLite model")
            elif os.path.exists('model/skin_cancer_model.h5'):
                from tensorflow.keras.models import load_model
                self.model = load_model('model/skin_cancer_model.h5')
                print("Loaded H5 model")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def create_ui(self):
        header_frame = tk.Frame(self.root, bg="#16213e", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title = tk.Label(
            header_frame,
            text="MOBILE SKIN ANALYZER",
            font=("Helvetica", 24, "bold"),
            bg="#16213e",
            fg="#00d9ff"
        )
        title.pack(pady=20)
        
        subtitle = tk.Label(
            header_frame,
            text="AI-Powered Skin Cancer Detection",
            font=("Helvetica", 10),
            bg="#16213e",
            fg="#888888"
        )
        subtitle.pack()
        
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        self.image_frame = tk.Frame(main_frame, bg="#16213e", width=300, height=300)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        placeholder = tk.Label(
            self.image_frame,
            text="Click 'Upload' to\nselect an image",
            font=("Helvetica", 14),
            bg="#16213e",
            fg="#555555"
        )
        placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.image_label = tk.Label(self.image_frame, bg="#16213e")
        self.image_label.pack(expand=True)
        
        btn_frame = tk.Frame(main_frame, bg="#1a1a2e")
        btn_frame.pack(pady=15)
        
        upload_btn = tk.Button(
            btn_frame,
            text="  UPLOAD IMAGE  ",
            command=self.upload_image,
            font=("Helvetica", 12, "bold"),
            bg="#0f3460",
            fg="white",
            padx=30,
            pady=12,
            relief=tk.FLAT,
            cursor="hand2",
            activebackground="#1a4a80",
            activeforeground="white"
        )
        upload_btn.grid(row=0, column=0, padx=10)
        
        self.analyze_btn = tk.Button(
            btn_frame,
            text="   ANALYZE   ",
            command=self.analyze_image,
            font=("Helvetica", 12, "bold"),
            bg="#00d9ff",
            fg="#1a1a2e",
            padx=30,
            pady=12,
            relief=tk.FLAT,
            cursor="hand2",
            state=tk.DISABLED,
            activebackground="#00b8e6",
            activeforeground="#1a1a2e"
        )
        self.analyze_btn.grid(row=0, column=1, padx=10)
        
        self.result_frame = tk.Frame(main_frame, bg="#16213e", width=500, height=180)
        self.result_frame.pack(pady=10, fill=tk.BOTH)
        self.result_frame.pack_propagate(False)
        
        self.result_title = tk.Label(
            self.result_frame,
            text="Upload an image and click ANALYZE",
            font=("Helvetica", 16, "bold"),
            bg="#16213e",
            fg="#888888"
        )
        self.result_title.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
        
        self.result_confidence = tk.Label(
            self.result_frame,
            text="",
            font=("Helvetica", 12),
            bg="#16213e",
            fg="#555555"
        )
        self.result_confidence.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
        
        self.result_note = tk.Label(
            self.result_frame,
            text="",
            font=("Helvetica", 10),
            bg="#16213e",
            fg="#666666",
            wraplength=450,
            justify=tk.CENTER
        )
        self.result_note.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
        
        info_frame = tk.Frame(self.root, bg="#16213e", height=40)
        info_frame.pack(fill=tk.X)
        info_frame.pack_propagate(False)
        
        info_label = tk.Label(
            info_frame,
            text="Lightweight CNN Model | TFLite Optimized | For Educational Use Only",
            font=("Helvetica", 9),
            bg="#16213e",
            fg="#444444"
        )
        info_label.pack(pady=10)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            
            img = Image.open(file_path)
            img.thumbnail((260, 260))
            self.processed_image = ImageTk.PhotoImage(img)
            
            for widget in self.image_frame.winfo_children():
                widget.destroy()
            
            self.image_label = tk.Label(self.image_frame, image=self.processed_image, bg="#16213e")
            self.image_label.image = self.processed_image
            self.image_label.pack(expand=True)
            
            self.analyze_btn.config(state=tk.NORMAL, bg="#00d9ff")
            self.result_title.config(text="Ready to analyze", fg="#888888")
            self.result_confidence.config(text="")
            self.result_note.config(text="")
    
    def analyze_image(self):
        if not self.current_image_path:
            return
        
        self.analyze_btn.config(state=tk.DISABLED, bg="#444444")
        self.result_title.config(text="Analyzing...", fg="#00d9ff")
        self.result_confidence.config(text="")
        self.result_note.config(text="")
        self.root.update()
        
        thread = threading.Thread(target=self._analyze_thread)
        thread.start()
    
    def _analyze_thread(self):
        try:
            img = image.load_img(self.current_image_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            if self.interpreter:
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                self.interpreter.set_tensor(input_details[0]['index'], img_array)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(output_details[0]['index'])
            else:
                prediction = self.model.predict(img_array)
            
            confidence = float(prediction[0][0])
            
            self.root.after(0, self._update_result, confidence)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
    
    def _update_result(self, confidence):
        is_malignant = confidence > 0.5
        
        if is_malignant:
            self.result_title.config(text="MALIGNANT (Cancerous)", fg="#ff4757")
            self.result_confidence.config(
                text=f"Confidence: {confidence:.1%}",
                fg="#ff4757"
            )
            self.result_note.config(
                text="This lesion shows signs of potential malignancy.\nPlease consult a dermatologist for professional diagnosis.",
                fg="#ff6b7a"
            )
        else:
            self.result_title.config(text="BENIGN (Non-Cancerous)", fg="#2ed573")
            self.result_confidence.config(
                text=f"Confidence: {(1-confidence):.1%}",
                fg="#2ed573"
            )
            self.result_note.config(
                text="This lesion appears to be benign (NOT cancer).\nHowever, consult a dermatologist for confirmation.",
                fg="#7bed9f"
            )
        
        self.analyze_btn.config(state=tk.NORMAL, bg="#00d9ff")
    
    def _show_error(self, error_msg):
        self.result_title.config(text="Error", fg="#ff4757")
        self.result_confidence.config(text=error_msg, fg="#ff4757")
        self.result_note.config(text="")
        self.analyze_btn.config(state=tk.NORMAL, bg="#00d9ff")
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ModernSkinAnalyzer()
    app.run()