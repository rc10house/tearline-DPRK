from PIL import Image 
import os

orig_img_width = 8192  
orig_img_height = 6386
small_img_width = 64
small_img_height = 64

if __name__=="__main__":
    files = os.listdir("./results/")
    # sort is essential to make sure files stay in correct order
    files.sort()

    horizontal_imgs = []
    img = Image.open("./results/" + files[0])

    for i, f in enumerate(files):
        if i==0: continue 
        new_img = Image.open("./results/" + f)
        img_width = img.size[0]

        if img_width + small_img_width > orig_img_width:
            horizontal_imgs.append(img)
            img = Image.open("./results/" + f)
            continue
        result = Image.new('RGB', (img_width+small_img_width, small_img_height))
        result.paste(im=img, box=(0, 0))
        result.paste(im=new_img, box=(img.size[0], 0))
        img = result

    # join horizontals 
    img = horizontal_imgs[0]
    for i in horizontal_imgs:
        if i == img: continue 
        result = Image.new('RGB', (img.size[0], img.size[1]+i.size[1]))
        result.paste(im=img, box=(0,0))
        result.paste(im=i, box=(0, img.size[1]))
        img = result

    img.save("merged.jpeg")

    
