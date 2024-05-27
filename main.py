import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tkinter import Tk, Label, Button, filedialog, Frame, PhotoImage
from tkinter import ttk
from PIL import Image, ImageOps, ImageTk

model = tf.keras.models.load_model("best_model.hdf5")


def load_image(file_path):
    """Load an image and preprocess it to match the input shape of the model."""
    img = Image.open(file_path)
    img = ImageOps.fit(img, (256, 256))
    img = np.array(img) / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img[np.newaxis, ...]


def predict_and_visualize(img_array):
    """Predict the mask using the model and visualize the results."""
    pred_y = model.predict(img_array)
    pred_mask = (pred_y[0, :, :, 0] > 0.15).astype(
        np.uint8
    )  # Assuming binary segmentation

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(img_array[0])

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="gray")
    plt.show()


def open_file():
    """Open a file dialog to select an image file."""
    file_path = filedialog.askopenfilename(
        filetypes=[
            (
                "Image files",
                "*.jpg;*.png;*.jpeg" ".tif",
            )
        ]
    )
    if file_path:
        img_array = load_image(file_path)
        predict_and_visualize(img_array)
        # Display the selected image in the UI
        img = Image.open(file_path)
        img.thumbnail((150, 150))  # Resize for thumbnail
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img


# Create the main window
root = Tk()
root.title("UNet Image Segmentation")
root.geometry("400x300")
root.resizable(False, False)
icon = PhotoImage(file="log.png")
root.iconphoto(False, icon)

# Style configuration
style = ttk.Style(root)
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.configure("TLabel", font=("Helvetica", 12), padding=10)

# Create a frame for better layout management
frame = Frame(root, padx=20, pady=20)
frame.pack(expand=True)

label = ttk.Label(frame, text="Select an image file to predict its mask:")
label.pack(pady=10)

open_button = ttk.Button(frame, text="Open Image", command=open_file)
open_button.pack(pady=20)

# Label to show the selected image
img_label = Label(frame)
img_label.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
