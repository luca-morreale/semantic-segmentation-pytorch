
def generate_image_name(input_image_name):
    img_name = input_image_name.split('/')[-1]
    return ''.join(img_name.split('.')[:-1]) + '_rgb.png', ''.join(img_name.split('.')[:-1]) + '.png'


