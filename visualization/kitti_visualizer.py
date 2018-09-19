import os
import numpy as np
from lib.utils.utils import unique
from visualization.utils_name_generation import generate_image_name
import cv2

colormap = {
    0: (128, 128, 128),    # Sky
    1: (128, 0, 0),        # Building
    2: (128, 64, 128),     # Road
    3: (0, 0, 192),        # Sidewalk
    4: (64, 64, 128),      # Fence
    5: (128, 128, 0),      # Vegetation
    6: (192, 192, 128),    # Pole
    7: (64, 0, 128),       # Car
    8: (192, 128, 128),    # Sign
    9: (64, 64, 0),        # Pedestrian
    10: (0, 128, 192),     # Cyclist
    11: (0, 0, 0)          # Void
}

conversion_list = {
    1:   1, #wall
    2:   1, #building;edifice
    3:   0, #sky
    4:   2, #floor;flooring
    5:   5, #tree
    6:   1, #ceiling
    7:   2, #road;route
    8:   11, #bed
    9:   1, #windowpane;window
    10:  5, #grass
    11:  11, #cabinet
    12:  3, #sidewalk;pavement
    13:  9, #person;individual;someone;somebody;mortal;soul
    14:  2, #earth;ground
    15:  1, #door;double;door
    16:  11, #table
    17:  11, #mountain;mount
    18:  5, #plant;flora;plant;life
    19:  11, #curtain;drape;drapery;mantle;pall
    20:  11, #chair
    21:  7, #car;auto;automobile;machine;motorcar
    22:  11, #water
    23:  11, #painting;picture
    24:  11, #sofa;couch;lounge
    25:  11, #shelf
    26:  1, #house
    27:  11, #sea
    28:  11, #mirror
    29:  11, #rug;carpet;carpeting
    30:  2, #field
    31:  11, #armchair
    32:  11, #seat
    33:  4, #fence;fencing
    34:  11, #desk
    35:  11, #rock;stone
    36:  11, #wardrobe;closet;press
    37:  6, #lamp
    38:  11, #bathtub;bathing;tub;bath;tub
    39:  4, #railing;rail
    40:  11, #,cushion
    41:  11, #base;pedestal;stand
    42:  11, #box
    43:  6, #column;pillar
    44:  8, #signboard;sign
    45:  11, #chest;of;drawers;chest;bureau;dresser
    46:  11, #counter
    47:  2, #sand
    48:  11, #sink
    49:  1, #skyscraper
    50:  11, #fireplace;hearth;open;fireplace
    51:  11, #refrigerator;icebox
    52:  11, #grandstand;covered;stand
    53:  2, #,path
    54:  4, #stairs;steps
    55:  2, #runway
    56:  1, #case;display;case;showcase;vitrine
    57:  11, #pool;table;billiard;table;snooker;table
    58:  11, #pillow
    59:  11, #screen;door;screen
    60:  4, #stairway;staircase
    61:  11, #river
    62:  11, #,bridge;span
    63:  11, #bookcase
    64:  11, #blind;screen
    65:  11, #coffee;table;cocktail;table
    66:  11, #toilet;can;commode;crapper;pot;potty;stool;throne
    67:  11, #flower
    68:  11, #book
    69:  11, #hill
    70:  11, #bench
    71:  11, #countertop
    72:  11, #stove;kitchen;stove;range;kitchen;range;cooking;stove
    73:  11, #palm;palm;tree
    74:  11, #kitchen;island
    75:  11, #computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system
    76:  11, #swivel;chair
    77:  11, #boat
    78:  11, #bar
    79:  11, #arcade;machine
    80:  11, #hovel;hut;hutch;shack;shanty
    81:  7, #bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle
    82:  11, #towel
    83:  6, #light;light;source
    84:  7, #truck;motortruck
    85:  1, #tower
    86:  11, #chandelier;pendant;pendent
    87:  11, #awning;sunshade;sunblind
    88:  6, #streetlight;street;lamp
    89:  11, #booth;cubicle;stall;kiosk
    90:  11, #television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box
    91:  11, #airplane;aeroplane;plane
    92:  11, #dirt;track
    93:  11, #apparel;wearing;apparel;dress;clothes
    94:  6, #pole
    95:  3, #land;ground;soil
    96:  11, #bannister;banister;balustrade;balusters;handrail
    97:  11, #escalator;moving;staircase;moving;stairway
    98:  11, #ottoman;pouf;pouffe;puff;hassock
    99:  11, #bottle
    100: 11, #buffet;counter;sideboard
    101: 11, #poster;posting;placard;notice;bill;card
    102: 11, #stage
    103: 7, #van
    104: 11, #ship
    105: 11, #fountain
    106: 11, #conveyer;belt;conveyor;belt;conveyer;conveyor;transporter
    107: 11, #canopy
    108: 11, #washer;automatic;washer;washing;machine
    109: 11, #plaything;toy
    110: 11, #swimming;pool;swimming;bath;natatorium
    111: 11, #0,stool
    112: 11, #barrel;cask
    113: 11, #basket;handbasket
    114: 11, #waterfall;falls
    115: 11, #tent;collapsible;shelter
    116: 11, #bag
    117: 10, #minibike;motorbike
    118: 11, #cradle
    119: 11, #oven
    120: 11, #ball
    121: 11, #food;solid;food
    122: 11, #step;stair
    123: 7, #tank;storage;tank
    124: 11, #trade;name;brand;name;brand;marque
    125: 11, #microwave;microwave;oven
    126: 11, #pot;flowerpot
    127: 11, #animal;animate;being;beast;brute;creature;fauna
    128: 10, #bicycle;bike;wheel;cycle
    129: 11, #lake
    130: 11, #dishwasher;dish;washer;dishwashing;machine
    131: 11, #screen;silver;screen;projection;screen
    132: 11, #blanket;cover
    133: 11, #sculpture
    134: 11, #hood;exhaust;hood
    135: 11, #sconce
    136: 11, #vase
    137: 8, #traffic;light;traffic;signal;stoplight
    138: 11, #tray
    139: 11, #ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin
    140: 11, #fan
    141: 11, #pier;wharf;wharfage;dock
    142: 11, #crt;screen
    143: 11, #plate
    144: 11, #monitor;monitoring;device
    145: 11, #bulletin;board;notice;board
    146: 11, #shower
    147: 11, #radiator
    148: 11, #glass;drinking;glass
    149: 11, #clock
    150: 11, #flag
}


def convert_labels_to_kitti(predictions, mode='BGR'):
    predictions = predictions.astype('int')
    labelmap_kitti = np.zeros(predictions.shape, dtype=np.uint8)
    labelmap_rgb = np.zeros((predictions.shape[0], predictions.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(predictions):
        if label < 0:
            continue

        label_kitti = conversion_list[label + 1]

        labelmap_rgb += (predictions == label)[:, :, np.newaxis] * \
            np.tile(np.uint8(colormap[label_kitti]),
                    (predictions.shape[0], predictions.shape[1], 1))
        labelmap_kitti[predictions == label] = label_kitti

    if mode == 'BGR':
        return labelmap_kitti, labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_kitti, labelmap_rgb


def visualize_result(data, preds, args):
    (img, info) = data

    kitti_pred, pred_color = convert_labels_to_kitti(preds)

    # aggregate images and save
    im_vis = pred_color.astype(np.uint8)

    img_name_rgb, img_name = generate_image_name(info)
    a = os.path.join(args.output_path, img_name_rgb)
    print(a)
    cv2.imwrite(os.path.join(args.output_path, img_name_rgb), im_vis)

    # aggregate images and save
    im_vis = kitti_pred.astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_path, img_name), im_vis)

