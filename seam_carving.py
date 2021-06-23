# ----- Import  necessary libraries -----
import cv2
import numpy as np 

from numba import jit
from scipy import ndimage as ndi

ENERGY_MASK_CONST = 100000.0             
MASK_THRESHOLD = 10                 # Intensity threshold for binary mask

# ----- Energy functions -----
# Backward energy
def backward_energy(img):
    pass

# Forward energy
def forward_energy(img):
    h, w = img.shape[:2]
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))

    U = np.roll(img, 1, axis=0)
    L = np.roll(img, 1, axis=1)
    R = np.roll(img, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)     
        
    return energy

# ----- Seam helper functions -----
# Minimum seam
@jit
def get_minimum_seam(img, rmask=None):
    h, w = img.shape[:2]

    M = forward_energy(img)

    # Areas under the masked region are weighted with a very high negative value to guarantee that the minimum seam will be routed through the masked region
    if rmask is not None:
        M[np.where(rmask > MASK_THRESHOLD)] = -ENERGY_MASK_CONST * 100

    backtrack = np.zeros_like(M, dtype=np.int) 

    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy
    
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool)
    j = np.argmin(M[-1])

    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()

    return np.array(seam_idx), boolmask

# ----- Seams removal -----
# Remove a seam
@jit
def remove_seam(img, boolmask):
    h, w = img.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return img[boolmask3c].reshape((h, w - 1, 3))

@jit
def remove_seam_gsc(img, boolmask): # Used for binary masks
    h, w = img.shape[:2]
    return img[boolmask].reshape((h, w - 1))

# Seams removal
def seams_removal(img, num_remove):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(img)

        img = remove_seam(img, boolmask)

    return img

# ----- Seams insertion ----- (Error)
# Add a seam
@jit
def add_seam(img, seam_idx):
    h, w = img.shape[:2]
    output = np.zeros((h, w + 1, 3), dtype=np.uint8) # Create output template, uint8 require for array to proper image

    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                # p
                p = np.average(img[row, col: col + 2, ch])
                output[row, col, ch] = img[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = img[row, col:, ch]
            else:
                p = np.average(img[row, col - 1: col + 1, ch])
                output[row, : col, ch] = img[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = img[row, col:, ch]

    return output

# Add a seam gsc
@jit
def add_seam_grayscale(img, seam_idx): # Used for binary masks   
    h, w = img.shape[:2]
    output = np.zeros((h, w + 1))
    for row in range(h):
        col = seam_idx[row]
        if col == 0:
            p = np.average(img[row, col: col + 2])
            output[row, col] = img[row, col]
            output[row, col + 1] = p
            output[row, col + 1:] = img[row, col:]
        else:
            p = np.average(img[row, col - 1: col + 1])
            output[row, : col] = img[row, : col]
            output[row, col] = p
            output[row, col + 1:] = img[row, col:]

    return output

# Seams insertion
def seams_insertion(img, num_add):
    seams_record = []
    temp_img = img.copy()

    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_img)
        seams_record.append(seam_idx)
        temp_img = remove_seam(temp_img, boolmask)

    seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        img = add_seam(img, seam)
        # update the remaining seam indices
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2         

    return img

# ----- Object removal -----
def object_removal(img, rmask, horizontal=None): 
    """ 
    rmask: path to the binary mask of the object that need to be removed 
    horizontal: enable when remove object horizontally    
    mask: protective mask for no carving regions ()
    """
    img = img.astype(np.float64)
    rmask = rmask.astype(np.float64)

    output = img

    h, w = img.shape[:2]

    if horizontal:
        output = np.rot90(output, 1)
        rmask = np.rot90(rmask, 1)

    while len(np.where(rmask > MASK_THRESHOLD)[0]) > 0:
        seam_idx, boolmask = get_minimum_seam(output, rmask)
                 
        output = remove_seam(output, boolmask)
        rmask = remove_seam_gsc(rmask, boolmask)

    num_add = (h if horizontal else w) - output.shape[1]
    output = seams_insertion(output, num_add)

    if horizontal:
        output = np.rot90(output, -1)

    return output
    
# ----- Mask functions -----
# Select ROI (Region of interest)
def bounding_roi():
    pass

def any_roi():
    pass
    
# Mask of object that needs removing
def remove_mask():
    pass

# Mask of regions that need protecting 
def protective_mask():
    pass

# Generate since Remove mode activated

# ----- MAIN CARVE FUNCTION -----
def seam_carving(img, remove=None, ):
    pass

# Test zone
img = cv2.imread('balloon.jpg', 1)
print(img.shape)
img = cv2.resize(img, (0,0), fx=0.3, fy=0.3) 
rmask = cv2.imread('balloon_mask.png', 0)
rmask = cv2.resize(rmask, (0,0), fx=0.3, fy=0.3) 

#seam_idx, boolmask = get_minimum_seam(img)
res = object_removal(img, rmask)

res = cv2.resize(res, (0,0), fx=3, fy=3) 
print(res.shape)
#print(seam_idx.shape, img.shape)

cv2.imshow('org', img)
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
