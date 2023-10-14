from pathlib import Path

from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog, ttk
import math

from image import *



class PageController():

    def __init__(self, controller) -> None:
        self.controller = controller

    def switch_hide_image_page(self):
        self.controller.show_frame("HideImagePage")

    def switch_reveal_image_page(self):
        self.controller.show_frame("RevealImagePage")

    def switch_load_model_page(self):
        self.controller.show_frame("LoadModelPage")


class HideImagePageController(PageController):

    def __init__(self, controller, cover, secret) -> None:
        super().__init__(controller)
        self.cover = cover
        self.secret = secret
    
    def cover_secret_image_picker(self, canvas, image_on_canvas, imagelist, is_cover):
        # global page1_cover, page1_secret
        img = image_picker(canvas, image_on_canvas, imagelist)
        
        if is_cover:
            self.cover = img
        else:
            self.secret = img

    def hide_image_button_command(self):
        StegaImageProcessing.hide_image(model=model, cover_o=self.cover, secret_o=self.secret)


class RevealImagePageController(PageController):

    def __init__(self, controller, secret) -> None:
        super().__init__(controller)
        self.secret = secret

    def secret_image_picker(self, canvas, image_on_canvas, imagelist):
        # global page2_cover
        img = image_picker(canvas, image_on_canvas, imagelist)
        self.secret = img

    def reveal_image_button_command(self):
        StegaImageProcessing.reveal_image(model=model, cover_o=self.secret, secret_o=self.secret)


class LoadModelPageController(PageController):

    def __init__(self, controller) -> None:
        super().__init__(controller)
        self.model = None

    def model_picker(self, entry):
        path=filedialog.askopenfilename(filetypes=[("Image File",'*.pth')])

        entry.delete(0,END)
        entry.insert(0,path)

        global model

        model = StegaImageProcessing.get_model(Path(path))
        self.model = model
        return path, model
        # print(model)


def image_picker(canvas, image_on_canvas, imagelist):
    path=filedialog.askopenfilename(filetypes=[("Image File",'*.JPEG *.png')])

    with Image.open(path) as img_thumbnail:
        width = 200
        height = 200
        imagex, imagey = img_thumbnail.size
        old_aspect = float(imagex)/float(imagey)
        new_aspect = float(width)/float(height)
        if old_aspect < new_aspect:
            height = math.ceil(width / old_aspect)
        else:
            width = math.ceil(height * old_aspect)

        image_display_size = width, height

        img_thumbnail.thumbnail(image_display_size, Image.Resampling.LANCZOS)
        
        
        imagex, imagey = img_thumbnail.size
        # print(img_thumbnail.size)
        if (imagex >= 200 and imagey >= 200):
            left = (imagex - 200)/2
            top = (imagey - 200)/2
            right = (imagex + 200)/2
            bottom = (imagey + 200)/2
            # Crop the center of the image
            img_thumbnail = img_thumbnail.crop((left, top, right, bottom))

        imagelist.append(ImageTk.PhotoImage(img_thumbnail))
        canvas.itemconfig(image_on_canvas, image = imagelist[-1])
    
    with Image.open(path) as img:
        return img.convert('RGB')
