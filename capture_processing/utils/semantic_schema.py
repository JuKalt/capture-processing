"""Versioned Sapiens2 dome29 labels; IDs describe anatomy, never screen side."""
FINE = ('background', 'apparel', 'eyeglass', 'face_neck', 'hair', 'left_foot',
        'left_hand', 'left_lower_arm', 'left_lower_leg', 'left_shoe', 'left_sock',
        'left_upper_arm', 'left_upper_leg', 'lower_clothing', 'right_foot',
        'right_hand', 'right_lower_arm', 'right_lower_leg', 'right_shoe',
        'right_sock', 'right_upper_arm', 'right_upper_leg', 'torso',
        'upper_clothing', 'lower_lip', 'upper_lip', 'lower_teeth', 'upper_teeth', 'tongue')
GENERAL = ('background', 'head_hair', 'exposed_body', 'clothing', 'footwear', 'accessories')
# Head includes face/neck. Socks belong to footwear. Apparel is unspecified accessories.
FINE_TO_GENERAL = (0, 5, 5, 1, 1, 2, 2, 2, 2, 4, 4, 2, 2, 3, 2, 2, 2, 2, 4, 4, 2, 2, 2, 3, 1, 1, 1, 1, 1)
UNKNOWN = 255
VERSION = 1


def schema():
    return {'version': VERSION, 'fine': dict(enumerate(FINE)),
            'general': dict(enumerate(GENERAL)), 'fine_to_general': list(FINE_TO_GENERAL),
            'unknown': UNKNOWN, 'identity': 'one captured subject; shared across cameras',
            'labels': 'visible surfaces; anatomical left/right; automatic pseudo-labels',
            'confidence': 'uint16 PNG divided by 65535; zero for unknown, one for capture background'}
