from math import log10, sqrt
import numpy as np

def preprocess_img(img):
    img = img - img.min()
    img = img / img.max()
    img = (img * 255).astype(np.uint8)
    return img


def cal_PSNR(true_img, fake_img):
    true_img = preprocess_img(true_img)
    fake_img = preprocess_img(fake_img)
    MSE = np.mean((true_img, fake_img) ** 2)
    if MSE == 0:
        print("@@@ Same images when calculating PSNR")
    max_intense = 255.0
    PSNR = 20 * log10(max_intense / sqrt(MSE))
    return PSNR


def pixelwise_accuracy_rgb(true_img, fake_img, thresh):

""" 
    calculate the pixelwise accuracy for the rgb
"""
    diffR = torch.abs(torch.round(true_img[..., 0])- torch.round(fake_img[..., 0]))
    diffG = torch.abs(torch.round(true_img[..., 1])- torch.round(fake_img[..., 1]))
    diffB = torch.abs(torch.round(true_img[..., 2]), torch.round(fake_img[..., 2]))

    predR = torch.less_equal(diffR, thresh).float()
    predG = torch.less_equal(diffG, thresh).float()
    predB = torch.less_equal(diffB, thresh).float()

    pred = predR * predG * predB

    return torch.mean(pred)

def evaluate_batch(true_imgs, fake_imgs, type = "pixel_rgb", thresh = 255*0.05):
"""
    calculate the batch and return mean, normally the batch size is

"""
    N = true_imgs.shape[0]
    accu = []
    for i in range(N):
        accu.append(pixelwise_accuracy_rgb(true_imgs[i], fake_imgs[i], thresh))

    return np.mean(accu)










# ----------
# PSNR
# ----------
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# ----------
# SSIM
# ----------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

