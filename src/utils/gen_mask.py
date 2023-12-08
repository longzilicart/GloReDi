# basic mask

import torch
import numpy as np



def sector_mask_np(shape, centre, radius, angle_range):
    """
    from : https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """
    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)
    (inner_radius, outer_radius) = radius
    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi
    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy) 
    theta = np.arctan2(x-cx,y-cy) - tmin
    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)
    # circular mask
    circmask_outer = r2 < outer_radius*outer_radius 
    circmask_inner = r2 >= inner_radius*inner_radius
    circmask = circmask_outer * circmask_inner
    # angular mask
    anglemask = theta <= (tmax-tmin)
    return circmask*anglemask

def gen_circle_mask_np(x, start=None, end=None, center = (0,0)):

    # 1. parse center and shape
    (center_x, center_y) = center
    width, height = x.size(-2), x.size(-1)
    mask = torch.zeros((width, height))
    # start radius and end radius
    if start is not None:
        r_start = start
    else:
        r_start = 0
    if end is not None:
        r_end = end
    else:
        r_end = np.sqrt(width**2 + height**2) #
    # generate circle mask
    circle_mask_np = sector_mask_np((width, height), (center_x, center_y), (r_start,r_end), (0,360))
    mask = torch.from_numpy(circle_mask_np.astype(int))
    return mask

def gen_square_mask_np(x, start=None, end=None, center=(0,0)):
    (center_x, center_y) = center
    width, height = x.size(-2), x.size(-1)
    mask = torch.zeros((width, height))
    
    start, end = start / 2 * 1.414, end / 2 * 1.414
    #
    if start is not None:
        s_start = start
    else:
        s_start = 0
    if end is not None:
        s_end = end
    else:
        s_end = max(width, height)
    
    # 
    y, x = np.ogrid[:width, :height]
    mask_inner = (x >= (center_x - s_start)) & (x <= (center_x + s_start)) & (y >= (center_y - s_start)) & (y <= (center_y + s_start))
    mask_outer = (x >= (center_x - s_end)) & (x <= (center_x + s_end)) & (y >= (center_y - s_end)) & (y <= (center_y + s_end))

    square_mask_np = np.logical_and(mask_outer, ~mask_inner) 
    mask = torch.from_numpy(square_mask_np.astype(int))
    return mask


# ----------- the pytorch version ------------


def sector_mask(shape, centre, radius, angle_range):
    """
    from : https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    pytorch version reproduced by longzilil
    """
    y,x = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
    cx,cy = centre
    tmin,tmax = torch.tensor(angle_range, dtype=torch.float32).mul_(3.14159265/180.0) 
    (inner_radius, outer_radius) = radius
    if tmax < tmin:
            tmax += 2*3.14159265
    r2 = (x-cx)**2 + (y-cy)**2
    theta = torch.atan2(x-cx,y-cy) - tmin
    theta %= (2*3.14159265)
    circmask_outer = r2 < outer_radius**2
    circmask_inner = r2 >= inner_radius**2
    circmask = circmask_outer & circmask_inner
    anglemask = theta <= (tmax-tmin)
    return circmask & anglemask

def gen_circle_mask(x, start=None, end=None, center = (0,0)):

    (center_x, center_y) = center
    width, height = x.size(-2), x.size(-1)
    mask = torch.zeros((width, height))
    if start is not None:
        r_start = start
    else:
        r_start = 0
    if end is not None:
        r_end = end
    else:
        r_end = torch.sqrt(width**2 + height**2).item()
    circle_mask = sector_mask((width, height), (center_x, center_y), (r_start,r_end), (0,360))
    mask = circle_mask.int()
    return mask

def gen_square_mask(x, start=None, end=None, center=(0,0)):

    (center_x, center_y) = center
    width, height = x.size(-2), x.size(-1)
    mask = torch.zeros((width, height))

    start, end = start / 2 * 1.414, end / 2 * 1.414
    if start is not None:
        s_start = start
    else:
        s_start = 0
    if end is not None:
        s_end = end
    else:
        s_end = max(width, height).item()

    y, x = torch.meshgrid(torch.arange(width), torch.arange(height))
    mask_inner = (x >= (center_x - s_start)) & (x <= (center_x + s_start)) & (y >= (center_y - s_start)) & (y <= (center_y + s_start))
    mask_outer = (x >= (center_x - s_end)) & (x <= (center_x + s_end)) & (y >= (center_y - s_end)) & (y <= (center_y + s_end))

    if s_start == 0: 
        square_mask = mask_outer
    else:
        square_mask = mask_outer & (~mask_inner)

    mask = square_mask.int()
    return mask

def gen_circle_mask_learn(x, start=None, end=None, center=(0,0)):
    (center_x, center_y) = center
    width, height = x.size(-2), x.size(-1)
    mask = torch.zeros((width, height))

    start, end = start / 2 * 1.414, end / 2 * 1.414
    if start is not None:
        s_start = start
    else:
        s_start = 0
    if end is not None:
        s_end = end
    else:
        s_end = max(width, height).item()

    y, x = torch.meshgrid(torch.arange(width, dtype=torch.float32), torch.arange(height, dtype=torch.float32))
    dist_to_center = ((x - center_x)**2 + (y - center_y)**2).sqrt().to(start.device)
    
    # tanh for soft edge, TODO modify to circle
    mask_inner = 0.5 * (torch.tanh(10 * (dist_to_center / 1.414 - s_start)) + 1)
    mask_outer = 0.5 * (torch.tanh(10 * (s_end - dist_to_center / 1.414)) + 1)
    mask = (mask_outer * mask_inner).clamp(0, 1)
    return mask



def gen_DCT2_circle_mask(x, start_ratio, end_ratio, center = (0,0)):
    # start, end = 0.0, 0.4
    assert (0 <= start_ratio <= 1) and (0 <= end_ratio <= 1)
    start_c, end_c = x.size(-2) * start_ratio * 1.414, x.size(-1) * end_ratio * 1.414
    mask = gen_circle_mask(x, start = start_c, end = end_c, center=center )
    return mask

def gen_DCT2_square_mask(x, start_ratio, end_ratio, center = (0,0)):
    # start, end = 0.0, 0.4
    assert (0 <= start_ratio <= 1) and (0 <= end_ratio <= 1)
    start_c, end_c = x.size(-2) * start_ratio * 1.414, x.size(-1) * end_ratio * 1.414
    mask = gen_square_mask(x, start = start_c, end = end_c, center=center )
    return mask

def gen_DCT2_circle_mask_learn(x, start_ratio, end_ratio, center = (0,0)):
    # start, end = 0.0, 0.4
    assert (0 <= start_ratio <= 1) and (0 <= end_ratio <= 1)
    start_c, end_c = x.size(-2) * start_ratio * 1.414, x.size(-1) * end_ratio * 1.414
    mask = gen_circle_mask_learn(x, start = start_c, end = end_c, center=center )
    return mask













if __name__ == '__main__':
    pass
