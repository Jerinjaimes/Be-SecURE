#importing the library
import tkinter as tk
from tkinter import *
import os
from PIL import Image, ImageTk

#Initializing Tkinter window
window= tk.Tk()
window.title("Be-SecURE")
window.geometry("1000x700")
my_font=("times",20,"bold")

def mask_detection():
    os.system('mask_detection.py')

def social_distance():
    os.system('social_distance.py')



# Setting up the Background
window["bg"]='black'
img = Canvas(window,width = 900,height=700)
img.pack()
image = ImageTk.PhotoImage(Image.open(r'Be-SecURE.jpg'))
img.create_image(20,6,image= image,anchor =NW)

# creating Button for Mask detection

button3 = tk.Button(window, text="Mask Detector", width=21,bg="#166FE5",font=my_font,fg= "black", command=mask_detection, height=2)
button3.place(x=180, y=600)

# creating Button for Social-distance detection
button4 = tk.Button(window, text="Social Distance Detector", width=21, bg="#166FE5",fg="black", command=social_distance,height=2,font=my_font)
button4.place(x=560,y=600)

window.mainloop()