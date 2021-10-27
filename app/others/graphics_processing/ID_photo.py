from PIL import Image
from removebg import RemoveBg
from matplotlib import colors
import numpy as np
# import dlib
api_key = 'fgm3fJnkbNGgTCHsgGN1Yq64'

def change_bg_color(api_key,file_in_path,color):
    if type(color) is str:
        color_list = np.multiply(colors.hex2color(colors.cnames[color]),255)
        color_list = map(int,color_list)
        color_tuple = tuple(color_list)
    elif type(color) is tuple:
        color_tuple = color
    else:
        print('UNKNOWN COLOR FORMAT.')
        return -1
    first_half_path, extension_name = file_in_path.split('.')
    remove_background = RemoveBg(api_key, 'error.log')
    remove_background.remove_background_from_img_file(file_in_path)
    file_on_background_path = '{}.{}_no_bg.png'.format(first_half_path, extension_name)
    file_on_background = Image.open(file_on_background_path)
    size_x, size_y = file_on_background.size
    file_output = Image.new('RGBA', file_on_background.size, color=color_tuple)
    file_output.paste(file_on_background,(0,0,size_x,size_y),file_on_background)
    file_output_path = '{}_output.png'.format(first_half_path)
    file_output.save(file_output_path)

# dlib 安装太麻烦暂时不做了
def face_recognition():
    print('face_recognition')

if __name__ == "__main__":
    file_base_path = 'D:\\PycharmProjects\\AIProcessingPlatform\\app\\others\\graphics_processing\\id_photos\\'
    file_name = '曹尹2.jpg'
    # color = (255, 255, 255)
    color = 'white'
    change_bg_color(api_key=api_key,file_in_path=file_base_path+file_name,color=color)