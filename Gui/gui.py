
from pathlib import Path

from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog, ttk
import math

from image import *

# Explicit imports to satisfy Flake8
# from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


ASSETS_PATH_PAGE1 = Path("Deep_Learning_Image_Steganography/Gui/assets/page1")
ASSETS_PATH_PAGE2 = Path("Deep_Learning_Image_Steganography/Gui/assets/page2")
ASSETS_PATH_PAGE3 = Path("Deep_Learning_Image_Steganography/Gui/assets/page3")


def relative_to_assets_page1(path: str) -> Path:
    return ASSETS_PATH_PAGE1 / Path(path)

def relative_to_assets_page2(path: str) -> Path:
    return ASSETS_PATH_PAGE2 / Path(path)

def relative_to_assets_page3(path: str) -> Path:
    return ASSETS_PATH_PAGE3 / Path(path)


class tkinterApp(Tk):
     
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
         
        # __init__ function for class Tk
        Tk.__init__(self, *args, **kwargs)
         
        # creating a container
        container = Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
  
        # initializing frames to an empty array
        self.frames = {} 
  
        self.frames["Page1"] = Page1(parent=container, controller=self)
        self.frames["Page2"] = Page2(parent=container, controller=self)
        self.frames["Page3"] = Page3(parent=container, controller=self)

        self.frames["Page1"].grid(row=0, column=0, sticky="nsew")
        self.frames["Page2"].grid(row=0, column=0, sticky="nsew")
        self.frames["Page3"].grid(row=0, column=0, sticky="nsew")
  
        self.show_frame("Page1")
  
    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class Page1(Frame):
     
    def __init__(self, parent, controller):
        self.controller = controller
        self.imagelist = []
        Frame.__init__(self, parent)
        self.canvas = Canvas(
            self,
            bg = "#3A7FF6",
            height = 519,
            width = 862,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )
        self.canvas.place(x=0, y=0)
        self.canvas.create_rectangle(
            209.0,
            0.0,
            869.0,
            519.0,
            fill="#FCFCFC",
            outline="")

        self.button_image_1 = PhotoImage(
            file=relative_to_assets_page1("button_1.png"))
        self.button_1 = Button(
            self,
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: page1_cover_secret_image_picker(self.canvas, self.image_1, self.imagelist, False),
            relief="flat"
        )
        self.button_1.place(
            x=586.0,
            y=117.0,
            width=136.0,
            height=24.0
        )
        self.button_image_2 = PhotoImage(
            file=relative_to_assets_page1("button_2.png"))
        self.button_2 = Button(
            self,
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: page1_hide_image_button_command(),
            relief="flat"
        )
        self.button_2.place(
            x=491.0,
            y=366.0,
            width=95.0,
            height=36.0
        )

        self.canvas.create_text(
            19.0,
            16.0,
            anchor="nw",
            text="Deep Learning \nImage \nSteganography",
            fill="#FCFCFC",
            font=("RobotoSlab Bold", 24 * -1)
        )

        self.button_image_3 = PhotoImage(
            file=relative_to_assets_page1("button_3.png"))
        self.button_3 = Button(
            self,
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: page1_cover_secret_image_picker(self.canvas, self.image_2, self.imagelist, True),
            relief="flat"
        )
        self.button_3.place(
            x=291.0,
            y=117.0,
            width=132.0,
            height=24.0
        )

        self.canvas.create_rectangle(
            20.0,
            160.0,
            176.0,
            165.0,
            fill="#FCFCFC",
            outline="")

        self.canvas.create_rectangle(
            20.0,
            208.0,
            176.0,
            213.0,
            fill="#FCFCFC",
            outline="")

        self.canvas.create_rectangle(
            19.0,
            301.0,
            175.0,
            306.0,
            fill="#FCFCFC",
            outline="")

        self.canvas.create_rectangle(
            20.0,
            254.0,
            176.0,
            259.0,
            fill="#FCFCFC",
            outline="")

        self.button_image_4 = PhotoImage(
            file=relative_to_assets_page1("button_4.png"))
        self.button_4 = Button(
            self,
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: switch_page1(self.controller),
            relief="flat"
        )
        self.button_4.place(
            x=20.0,
            y=166.0,
            width=156.0,
            height=41.0
        )

        self.button_image_5 = PhotoImage(
            file=relative_to_assets_page1("button_5.png"))
        self.button_5 = Button(
            self,
            image=self.button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: switch_page2(self.controller),
            relief="flat"
        )
        self.button_5.place(
            x=20.0,
            y=213.0,
            width=156.0,
            height=41.0
        )

        self.button_image_6 = PhotoImage(
            file=relative_to_assets_page1("button_6.png"))
        self.button_6 = Button(
            self,
            image=self.button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: switch_page3(self.controller),
            relief="flat"
        )
        self.button_6.place(
            x=19.0,
            y=260.0,
            width=156.0,
            height=41.0
        )

        self.image_image_1 = PhotoImage(
            file=relative_to_assets_page1("image_1.png"))
        self.image_1 = self.canvas.create_image(
            686.0,
            247.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(
            file=relative_to_assets_page1("image_2.png"))
        self.image_2 = self.canvas.create_image(
            391.0,
            249.0,
            image=self.image_image_2
        )

class Page2(Frame):
    
    def __init__(self, parent, controller):
        self.controller = controller
        self.imagelist = []
        Frame.__init__(self, parent)
        self.canvas = Canvas(
            self,
            bg = "#3A7FF6",
            height = 519,
            width = 862,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        self.canvas.place(x = 0, y = 0)
        self.canvas.create_rectangle(
            209.0,
            0.0,
            869.0,
            519.0,
            fill="#FCFCFC",
            outline="")

        self.button_image_1 = PhotoImage(
            file=relative_to_assets_page2("button_1.png"))
        self.button_1 = Button(
            self,
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: page2_reveal_image_button_command(),
            relief="flat"
        )
        self.button_1.place(
            x=491.0,
            y=366.0,
            width=95.0,
            height=36.0
        )

        self.canvas.create_text(
            19.0,
            16.0,
            anchor="nw",
            text="Deep Learning \nImage \nSteganography",
            fill="#FCFCFC",
            font=("RobotoSlab Bold", 24 * -1)
        )

        self.button_image_2 = PhotoImage(
            file=relative_to_assets_page2("button_2.png"))
        self.button_2 = Button(
            self,
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: page2_cover_image_picker(self.canvas, self.image_1, self.imagelist),
            relief="flat"
        )
        self.button_2.place(
            x=439.0,
            y=118.0,
            width=132.0,
            height=24.0
        )

        self.canvas.create_rectangle(
            20.0,
            160.0,
            176.0,
            165.0,
            fill="#FCFCFC",
            outline="")

        self.canvas.create_rectangle(
            20.0,
            208.0,
            176.0,
            213.0,
            fill="#FCFCFC",
            outline="")

        self.canvas.create_rectangle(
            19.0,
            301.0,
            175.0,
            306.0,
            fill="#FCFCFC",
            outline="")

        self.canvas.create_rectangle(
            20.0,
            254.0,
            176.0,
            259.0,
            fill="#FCFCFC",
            outline="")

        self.button_image_3 = PhotoImage(
            file=relative_to_assets_page2("button_3.png"))
        self.button_3 = Button(
            self,
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: switch_page1(self.controller),
            relief="flat"
        )
        self.button_3.place(
            x=20.0,
            y=166.0,
            width=156.0,
            height=41.0
        )

        self.button_image_4 = PhotoImage(
            file=relative_to_assets_page2("button_4.png"))
        self.button_4 = Button(
            self,
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: switch_page2(self.controller),
            relief="flat"
        )
        self.button_4.place(
            x=20.0,
            y=213.0,
            width=156.0,
            height=41.0
        )

        self.button_image_5 = PhotoImage(
            file=relative_to_assets_page2("button_5.png"))
        self.button_5 = Button(
            self,
            image=self.button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: switch_page3(self.controller),
            relief="flat"
        )
        self.button_5.place(
            x=19.0,
            y=260.0,
            width=156.0,
            height=41.0
        )

        self.image_image_1 = PhotoImage(
            file=relative_to_assets_page2("image_1.png"))
        self.image_1 = self.canvas.create_image(
            539.0,
            250.0,
            image=self.image_image_1
        )


class Page3(Frame):
     
    def __init__(self, parent, controller):
         
        Frame.__init__(self, parent)
        self.controller = controller
        self.canvas = Canvas(
            self,
            bg = "#3A7FF6",
            height = 519,
            width = 862,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        self.canvas.place(x = 0, y = 0)
        self.canvas.create_rectangle(
            209.0,
            0.0,
            869.0,
            519.0,
            fill="#FCFCFC",
            outline="")

        self.entry_image_1 = PhotoImage(
            file=relative_to_assets_page3("entry_1.png"))
        self.entry_bg_1 = self.canvas.create_image(
            538.5,
            256.5,
            image=self.entry_image_1
        )
        self.entry_1 = Entry(
            self,
            bd=0,
            bg="#F1F5FF",
            fg="#000716",
            highlightthickness=0
        )
        self.entry_1.place(
            x=378.0,
            y=226.0,
            width=321.0,
            height=59.0
        )

        self.canvas.create_text(
            19.0,
            16.0,
            anchor="nw",
            text="Deep Learning \nImage \nSteganography",
            fill="#FCFCFC",
            font=("RobotoSlab Bold", 24 * -1)
        )

        self.button_image_1 = PhotoImage(
            file=relative_to_assets_page3("button_1.png"))
        self.button_1 = Button(
            self,
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: model_picker(self.entry_1),
            relief="flat"
        )
        self.button_1.place(
            x=366.0,
            y=189.0,
            width=146.0,
            height=24.0
        )

        self.canvas.create_rectangle(
            20.0,
            160.0,
            176.0,
            165.0,
            fill="#FCFCFC",
            outline="")

        self.canvas.create_rectangle(
            20.0,
            208.0,
            176.0,
            213.0,
            fill="#FCFCFC",
            outline="")

        self.canvas.create_rectangle(
            19.0,
            301.0,
            175.0,
            306.0,
            fill="#FCFCFC",
            outline="")

        self.canvas.create_rectangle(
            20.0,
            254.0,
            176.0,
            259.0,
            fill="#FCFCFC",
            outline="")

        self.button_image_2 = PhotoImage(
            file=relative_to_assets_page3("button_2.png"))
        self.button_2 = Button(
            self,
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: switch_page1(self.controller),
            relief="flat"
        )
        self.button_2.place(
            x=20.0,
            y=166.0,
            width=156.0,
            height=41.0
        )

        self.button_image_3 = PhotoImage(
            file=relative_to_assets_page3("button_3.png"))
        self.button_3 = Button(
            self,
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: switch_page2(self.controller),
            relief="flat"
        )
        self.button_3.place(
            x=20.0,
            y=213.0,
            width=156.0,
            height=41.0
        )

        self.button_image_4 = PhotoImage(
            file=relative_to_assets_page3("button_4.png"))
        self.button_4 = Button(
            self,
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: switch_page3(self.controller),
            relief="flat"
        )
        self.button_4.place(
            x=19.0,
            y=260.0,
            width=156.0,
            height=41.0
        )


def switch_page1(controller):
    controller.show_frame("Page1")


def switch_page2(controller):
    controller.show_frame("Page2")


def switch_page3(controller):
    controller.show_frame("Page3")


def image_picker(canvas, image_on_canvas, imagelist):
    path=filedialog.askopenfilename(filetypes=[("Image File",'*.JPEG')])
    # path=filedialog.askopenfilename(filetypes=(('Image File', '*.JPEG'),))
    
    # # add `, initialdir="..."` to set the initial directory shown in the dialog
    # filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx .xls")])
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
    

def set_default_cover_secret_images():
    global page1_cover, page1_secret, page2_cover
    p1_cover = relative_to_assets_page1("image_1.png")
    p1_secret = relative_to_assets_page1("image_2.png")
    p2_cover = relative_to_assets_page2("image_1.png")

    with Image.open(p1_cover) as img:
        page1_cover = img.convert('RGB')

    with Image.open(p1_secret) as img:
        page1_secret = img.convert('RGB')

    with Image.open(p2_cover) as img:
        page2_cover = img.convert('RGB')


def page1_cover_secret_image_picker(canvas, image_on_canvas, self, is_cover):
    global page1_cover, page1_secret
    img = image_picker(canvas, image_on_canvas, self)
    
    if is_cover:
        page1_cover = img
    else:
        page1_secret = img


def page2_cover_image_picker(canvas, image_on_canvas, self):
    global page2_cover
    img = image_picker(canvas, image_on_canvas, self)
    page2_cover = img


def page1_hide_image_button_command():
    hide_image(model=model, cover_o=page1_cover, secret_o=page1_secret)


def page2_reveal_image_button_command():
    reveal_image(model=model, cover_o=page2_cover, secret_o=page2_cover)


def model_picker(entry):
    path=filedialog.askopenfilename(filetypes=[("Image File",'*.pth')])

    entry.delete(0,END)
    entry.insert(0,path)

    global model

    model = get_model(Path(path))
    return path, model
    # print(model)



# def open_popup():
#     top= Toplevel(win)
#     top.geometry("862x519")
#     top.title("Steganography Hide")
# #    Label(top, text= "Hello World!", font=('Mistral 18 bold')).place(x=150,y=80)

#     l = Label(win, text="Input")
#     l.grid(row=0, column=0)

#     b = ttk.Button(win, text="Okay", command=win.destroy)
#     b.grid(row=1, column=0)



win = tkinterApp()
# win = Tk()
set_default_cover_secret_images()

win.geometry("862x519")
win.configure(bg = "#3A7FF6")
win.title("Deep Learning Image Steganography")
win.resizable(False, False)
win.mainloop()
