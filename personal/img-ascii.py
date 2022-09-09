import math 
import random 
import PIL 
from PIL import Image
import requests
from io import BytesIO



#chars = ["$","E","F","L","l","v","!",";",",","."]
chars = ["#","0","L","J","m","k","q","z","v","i","_","+","~",","]

#url = "http://bit.ly/1UuWAmt"
url = r"https://en.wikipedia.org/wiki/Painting#/media/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg"
response = requests.get(url)

#img = Image.open(BytesIO(response.content))
img = Image.open(r"C:\Users\user\Downloads\painting.png")
img = img.convert('L')  
img = img.resize((300,150))
img = img.rotate(90)
for i in range(img.width):
    string1 = " "
    for x in range(img.height):
        col  = img.getpixel((i,x))
        col = col/255
        col = col*(len(chars)+1)
        col -= 0.5 
        col = int(col)
        #print(col)
        string1 = string1+chars[col]
    print(string1)
    














