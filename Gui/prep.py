
from pathlib import Path

from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog, ttk

from prep_controller import *
from image import *

# Explicit imports to satisfy Flake8
# from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


ASSETS_PATH_PAGE1 = Path("Deep_Learning_Image_Steganography/Gui/assets/HideImagePage")
ASSETS_PATH_PAGE2 = Path("Deep_Learning_Image_Steganography/Gui/assets/RevealImagePage")
ASSETS_PATH_PAGE3 = Path("Deep_Learning_Image_Steganography/Gui/assets/LoadModelPage")


def relative_to_assets_page1(path: str) -> Path:
    return ASSETS_PATH_PAGE1 / Path(path)

def relative_to_assets_page2(path: str) -> Path:
    return ASSETS_PATH_PAGE2 / Path(path)

def relative_to_assets_page3(path: str) -> Path:
    return ASSETS_PATH_PAGE3 / Path(path)


class GUI(Tk):
    '''
    The main graphical user interface. Provides functionally for users to pick images and steganography models to hide or reveal images with them.
    '''
     
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
        
        self.hide_page_controller = HideImagePageController(controller=self, cover=cover, secret=secret)
        self.reveal_page_controller = RevealImagePageController(controller=self, secret=secret)
        self.load_page_controller = LoadModelPageController(controller=self)
  
        self.frames["HideImagePage"] = HideImagePage(parent=container, controller=self.hide_page_controller)
        self.frames["RevealImagePage"] = RevealImagePage(parent=container, controller=self.reveal_page_controller)
        self.frames["LoadModelPage"] = LoadModelPage(parent=container, controller=self.load_page_controller)


        self.frames["HideImagePage"].grid(row=0, column=0, sticky="nsew")
        self.frames["RevealImagePage"].grid(row=0, column=0, sticky="nsew")
        self.frames["LoadModelPage"].grid(row=0, column=0, sticky="nsew")
  
        self.show_frame("HideImagePage")
  
    # to display the current frame passed as
    # parameter
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class HideImagePage(Frame):
     
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
            command=lambda: self.controller.cover_secret_image_picker(self.canvas, self.image_1, self.imagelist, False),
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
            command=lambda:self.controller.hide_image_button_command(),
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
            command=lambda: self.controller.cover_secret_image_picker(self.canvas, self.image_2, self.imagelist, True),
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
            command=lambda: self.controller.switch_hide_image_page(),
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
            command=lambda: self.controller.switch_reveal_image_page(),
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
            command=lambda: self.controller.switch_load_model_page(),
            relief="flat"
        )
        self.button_6.place(
            x=19.0,
            y=260.0,
            width=156.0,
            height=41.0
        )

        self.image_image_1 = PhotoImage(
            file=relative_to_assets_page1("image_2.png"))
        self.image_1 = self.canvas.create_image(
            686.0,
            247.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(
            file=relative_to_assets_page1("image_1.png"))
        self.image_2 = self.canvas.create_image(
            391.0,
            249.0,
            image=self.image_image_2
        )

        self.resize_warning_1 = self.canvas.create_text((686.0, 360.0),text="Image is resized to 224x224.", fill='black')
        self.resize_warning_2 = self.canvas.create_text((391.0, 360.0),text="Image is resized to 224x224.", fill='black')



class RevealImagePage(Frame):
    
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
            command=lambda: self.controller.reveal_image_button_command(),
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
            command=lambda: self.controller.secret_image_picker(self.canvas, self.image_1, self.imagelist),
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
            command=lambda: self.controller.switch_hide_image_page(),
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
            command=lambda: self.controller.switch_reveal_image_page(),
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
            command=lambda: self.controller.switch_load_model_page(),
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

        self.resize_warning_1 = self.canvas.create_text((539.0, 357.0),text="Image is resized to 224x224.", fill='black')


class LoadModelPage(Frame):
     
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
            command=lambda: self.controller.model_picker(self.entry_1),
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
            command=lambda: self.controller.switch_hide_image_page(),
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
            command=lambda: self.controller.switch_reveal_image_page(),
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
            command=lambda: self.controller.switch_load_model_page(),
            relief="flat"
        )
        self.button_4.place(
            x=19.0,
            y=260.0,
            width=156.0,
            height=41.0
        )


    

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



# def page2_cover_image_picker(canvas, image_on_canvas, self):
#     global page2_cover
#     img = image_picker(canvas, image_on_canvas, self)
#     page2_cover = img


# def page1_hide_image_button_command():
#     StegaImageProcessing.hide_image(model=model, cover_o=page1_cover, secret_o=page1_secret)


# def page2_reveal_image_button_command():
#     StegaImageProcessing.reveal_image(model=model, cover_o=page2_cover, secret_o=page2_cover)




# def open_popup():
#     top= Toplevel(win)
#     top.geometry("862x519")
#     top.title("Steganography Hide")
# #    Label(top, text= "Hello World!", font=('Mistral 18 bold')).place(x=150,y=80)

#     l = Label(win, text="Input")
#     l.grid(row=0, column=0)

#     b = ttk.Button(win, text="Okay", command=win.destroy)
#     b.grid(row=1, column=0)



if __name__ == "__main__":

    win = GUI()
    # win = Tk()
    set_default_cover_secret_images()

    win.geometry("862x519")
    win.configure(bg = "#3A7FF6")
    win.title("Deep Learning Image Steganography")
    win.resizable(False, False)
    win.mainloop()
