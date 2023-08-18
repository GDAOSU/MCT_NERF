import glob
import os

import cv2


def resize_imgs(in_dir,out_dir):
    img_dir=os.path.join(in_dir,"images")
    out_img_dir=os.path.join(out_dir,"images")
    #os.mkdir(out_img_dir)
    #process img
    # imgs=glob.glob(os.path.join(img_dir,"*.jpg"))
    # for img in imgs:
    #     img_name=os.path.basename(img)
    #     img_cv=cv2.imread(img)
    #     height,width,_=img_cv.shape
    #     img_resize=cv2.resize(img_cv,(int(width/2),int(height/2)))
    #     cv2.imwrite(os.path.join(out_img_dir,img_name),img_resize)
    
    #process colmap file
    sparse_dir=os.path.join(in_dir,"sparse")
    sparse_out_dir=os.path.join(out_dir,"sparse")
    os.mkdir(sparse_out_dir)
    in_data=open(os.path.join(sparse_dir,"cameras.txt"),'r')
    out_data=open(os.path.join(sparse_out_dir,"cameras.txt"),'w')
    lines=in_data.readlines()
    for id,line in enumerate(lines):
        if id==0:
            out_data.write(line)
        else:
            line=line.rstrip("\n")
            tokens=line.split(" ")
            out_line="{} {} {} {} {} {} {} {}\n".format(tokens[0],tokens[1],int(int(tokens[2])/2),int(int(tokens[3])/2),
                                                        float(tokens[4])/2,float(tokens[5])/2,float(tokens[6])/2,float(tokens[7])/2)
            out_data.write(out_line)
    in_data.close()
    out_data.close()
    






resize_imgs(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense',
            r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense_2')