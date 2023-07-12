import glob as glob
import math
import os

import cv2 as cv

COLMAP_PROJECT_PATH = r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\dense_offset'
# TILE_X = 4088
# TILE_Y = 3066
TILE_X = 800
TILE_Y = 800

path_camera_in = COLMAP_PROJECT_PATH + '/sparse/cameras.txt'
path_image_in = COLMAP_PROJECT_PATH + '/sparse/images.txt'
#path_qin_in = COLMAP_PROJECT_PATH + '/images/pose.qinv2'
path_img_in = COLMAP_PROJECT_PATH + '/images/'
#imglist = glob.glob(COLMAP_PROJECT_PATH + '/images/*.*[!qinv2|!qin]')

if not (os.path.exists(COLMAP_PROJECT_PATH+'/sparse_new')):
    os.mkdir(COLMAP_PROJECT_PATH+'/sparse_new')
if not (os.path.exists(COLMAP_PROJECT_PATH+'/images_new')):
    os.mkdir(COLMAP_PROJECT_PATH+'/images_new')

path_camera_out = COLMAP_PROJECT_PATH + '/sparse_new/cameras.txt'
path_image_out = COLMAP_PROJECT_PATH + '/sparse_new/images.txt'
#path_qin_out = COLMAP_PROJECT_PATH + '/images_new/pose.qinv2'
path_img_out = COLMAP_PROJECT_PATH+'/images_new/'

f_camera_out = open(path_camera_out,'w')
f_camera_in = open(path_camera_in,'r')

camera_id = 0
num_tiles = []
camera_accumulate = []
camera_accumulate.append(0)
split_dict = dict()
principle_dict = dict()
pond_num = 0

for line in f_camera_in.readlines():
    content = line.split()
    if len(content)==0:
        a=1
    elif content[0] == '#':
        pond_num += 1
        f_camera_out.write(line)
    else:
        width = int(content[2])
        height = int(content[3])
        num_tile_X = int(math.ceil(width/TILE_X))
        num_tile_Y = int(math.ceil(height/TILE_Y))
        num_tile = int(num_tile_X*num_tile_Y)
        num_tiles.append(num_tile)
        split_vector = []
        principle_vector = []
        for i in range(num_tile):
            tmp = []
            id_x = i%num_tile_X
            id_y = math.floor(i/num_tile_X)
            id_x_start = int(id_x*TILE_X)
            id_x_end = int((id_x+1)*TILE_X-1) if int((id_x+1)*TILE_X)<=width-1 else width-1
            id_y_start = int(id_y*TILE_Y)
            id_y_end = int((id_y+1)*TILE_Y-1) if int((id_y+1)*TILE_Y)<=height-1 else height-1
            f_camera_out.write("{} {} {} {} {} {} {} {}\n".format(camera_id,"PINHOLE",id_x_end-id_x_start+1,id_y_end-id_y_start+1,content[4],content[5],float(content[6])-id_x_start,float(content[7])-id_y_start))
            camera_id += 1
            tmp.append([id_x_start,id_y_start,id_x_end,id_y_end])
            split_vector.append(tmp)
            id_x_center = float(content[6]) - (id_x_start + (id_x_end-id_x_start+1)/2)
            id_y_center = -(float(content[7]) - (id_y_start + (id_y_end-id_y_start+1)/2))
            tmp1 = []
            tmp1.append([id_x_center,id_y_center])
            principle_vector.append(tmp1)
        camera_accumulate.append(camera_id)
        split_dict[int(content[0])] = split_vector
        principle_dict[int(content[0])] = principle_vector

f_camera_in.close()

if pond_num == 3:
    f_camera_out = open(path_camera_out,'r')
    data = f_camera_out.readlines()
    data[2] = "# Number of cameras: {}\n".format(camera_id)
    f_camera_out = open(path_camera_out,'w')
    f_camera_out.writelines(data)
f_camera_out.close()

f_image_out = open(path_image_out,'w')
f_image_in = open(path_image_in,'r')

image_id = 1
line_id = 1
img_cam = []
pond_num = 0
for line in f_image_in.readlines():
    content = line.split()
    if len(content)==0:
        a=1
    elif content[0] == '#':
        pond_num += 1
        f_image_out.write(line)
    elif pond_num == 4:
        if line_id%2==1:
            img = cv.imread(path_img_in+content[9],cv.IMREAD_UNCHANGED)
            cam_id = int(content[8]) -1 
            img_cam.append(cam_id)
            for i in range(num_tiles[cam_id]):
                content_new = content.copy()
                content_new[0] = image_id
                content_new[8] = camera_accumulate[cam_id]+i
                filename,extension = os.path.splitext(content_new[9])
                content_new[9] = filename+"_{:04d}".format(int(i))+extension
                for component in content_new:
                    f_image_out.write(str(component) + ' ')
                f_image_out.write('\n')
                image_id += 1
                f_image_out.write("\n")
                img_cut = img[split_dict[int(content[8])][i][0][1]:split_dict[int(content[8])][i][0][3]+1,split_dict[int(content[8])][i][0][0]:split_dict[int(content[8])][i][0][2]+1]
                cv.imwrite(path_img_out+content_new[9],img_cut)
    elif pond_num == 3:
        if line_id%2==0:
            img = cv.imread(path_img_in+content[9],cv.IMREAD_UNCHANGED)
            cam_id = int(content[8]) -1
            img_cam.append(cam_id)
            for i in range(num_tiles[cam_id]):
                content_new = content.copy()
                content_new[0] = image_id
                content_new[8] = camera_accumulate[cam_id]+i
                filename,extension = os.path.splitext(content_new[9])
                content_new[9] = filename+"_{:04d}".format(int(i))+extension
                for component in content_new:
                    f_image_out.write(str(component) + ' ')
                f_image_out.write('\n')
                image_id += 1
                f_image_out.write("\n")
                img_cut = img[split_dict[int(content[8])][i][0][1]:split_dict[int(content[8])][i][0][3]+1,split_dict[int(content[8])][i][0][0]:split_dict[int(content[8])][i][0][2]+1]
                cv.imwrite(path_img_out+content_new[9],img_cut)
    line_id+=1

f_image_in.close()
if pond_num == 4:
    f_image_out = open(path_image_out,'r')
    data = f_image_out.readlines()
    data[3] = "# Number of images: {}\n".format(image_id-1)
    f_image_out = open(path_image_out,'w')
    f_image_out.writelines(data)
f_image_out.close()

# f_qin_out = open(path_qin_out,'w')
# f_qin_in = open(path_qin_in,'r')

# line_id = 0
# f_qin_out.write(str(image_id-1)+'\n')
# for line in f_qin_in.readlines()[1:]:
#     content = line.split()
#     cam_id = img_cam[line_id]
#     for i in range(num_tiles[cam_id]):
#         content_new = content.copy()
#         filename,extension = os.path.splitext(content_new[0])
#         content_new[0] = filename+"_{:04d}".format(int(i))+extension
#         content_new[2] = principle_dict[int(cam_id)][i][0][0]
#         content_new[3] = principle_dict[int(cam_id)][i][0][1]
#         content_new[6] = split_dict[int(cam_id)][i][0][2]+1-split_dict[int(cam_id)][i][0][0]
#         content_new[7] = split_dict[int(cam_id)][i][0][3]+1-split_dict[int(cam_id)][i][0][1]
#         for component in content_new:
#             f_qin_out.write(str(component) + ' ')
#         f_qin_out.write('\n')
#     line_id+=1

# f_qin_in.close()
# f_qin_out = open(path_qin_out,'r')
# data = f_qin_out.readlines()
# data[0] = "{}\n".format(image_id-1)
# f_qin_out = open(path_qin_out,'w')
# f_qin_out.writelines(data)
# f_qin_out.close()