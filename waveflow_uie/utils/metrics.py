"""Standalone image quality metrics (no basicsr dependency).

Implements PSNR, SSIM, LPIPS, UCIQE, UIQM — matching basicsr's implementations
but importable without triggering basicsr's auto-import machinery.
"""

import cv2
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Color conversion helpers
# ---------------------------------------------------------------------------

def _bgr2y(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 HWC to Y channel (BT.601), return float64 [0, 255]."""
    img = img_bgr.astype(np.float64) / 255.0
    y = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    return y


def _rgb2y(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 HWC to Y channel (BT.601), return float64 [0, 255]."""
    img = img_rgb.astype(np.float64) / 255.0
    y = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    return y


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def calculate_psnr(
    img: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 2,
    test_y_channel: bool = True,
    input_order: str = 'HWC',
    is_bgr: bool = True,
) -> float:
    """Calculate PSNR between two uint8 HWC images.

    Args:
        img, img2: Images [0, 255] uint8.
        crop_border: Pixels to crop from each edge.
        test_y_channel: Compute on Y channel of YCbCr.
        input_order: 'HWC' or 'CHW'.
        is_bgr: If True, input is BGR; if False, input is RGB.
    """
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel and img.ndim == 3 and img.shape[2] == 3:
        # Convert to Y channel
        if is_bgr:
            img = np.dot(img / 255.0, [24.966, 128.553, 65.481]) + 16.0
            img2 = np.dot(img2 / 255.0, [24.966, 128.553, 65.481]) + 16.0
        else:
            img = np.dot(img / 255.0, [65.481, 128.553, 24.966]) + 16.0
            img2 = np.dot(img2 / 255.0, [65.481, 128.553, 24.966]) + 16.0

    mse = np.mean((img - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _ssim_single(img: np.ndarray, img2: np.ndarray) -> float:
    """SSIM for a single-channel image pair. Matches basicsr's Gaussian window."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def calculate_ssim(
    img: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 2,
    test_y_channel: bool = True,
    input_order: str = 'HWC',
    is_bgr: bool = True,
) -> float:
    """Calculate SSIM. Matches basicsr: per-channel SSIM then averaged."""
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel and img.ndim == 3 and img.shape[2] == 3:
        if is_bgr:
            img = np.dot(img / 255.0, [24.966, 128.553, 65.481]) + 16.0
            img2 = np.dot(img2 / 255.0, [24.966, 128.553, 65.481]) + 16.0
        else:
            img = np.dot(img / 255.0, [65.481, 128.553, 24.966]) + 16.0
            img2 = np.dot(img2 / 255.0, [65.481, 128.553, 24.966]) + 16.0
        img = img[..., None]
        img2 = img2[..., None]

    if img.ndim == 2:
        return _ssim_single(img, img2)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim_single(img[..., i], img2[..., i]))
    return float(np.mean(ssims))


# ---------------------------------------------------------------------------
# FID (Frechet Inception Distance)
# ---------------------------------------------------------------------------

_inception_model = None


def _get_inception_features_extractor(device):
    """Load InceptionV3 and return feature extractor that outputs 2048-dim pool3 features."""
    global _inception_model
    if _inception_model is None:
        import torch
        import torch.nn as nn
        from torchvision.models import inception_v3, Inception_V3_Weights

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = nn.Identity()  # output 2048-dim pool features
        net.eval()
        _inception_model = net.to(device)
    return _inception_model


class FIDCalculator:
    """Frechet Inception Distance calculator.

    Accumulates real and fake image features across batches, then computes FID.
    Use:
        fid = FIDCalculator(device)
        fid.update_real(real_uint8_rgb_batch)  # (N, H, W, 3) uint8
        fid.update_fake(fake_uint8_rgb_batch)
        score = fid.compute()
    """

    def __init__(self, device='cuda'):
        import torch
        self.device = device
        self.real_features = []
        self.fake_features = []
        self.model = _get_inception_features_extractor(device)

    def _extract(self, imgs_uint8_rgb):
        """Extract features from a batch of uint8 RGB images (N, H, W, 3)."""
        import torch
        import torch.nn.functional as F

        if imgs_uint8_rgb.ndim == 3:
            imgs_uint8_rgb = imgs_uint8_rgb[None, ...]

        # To tensor (N, 3, H, W), float32 in [0, 1]
        x = torch.from_numpy(imgs_uint8_rgb).float().permute(0, 3, 1, 2) / 255.0
        x = x.to(self.device)

        # Resize to 299x299 (InceptionV3 input size)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Normalize to [-1, 1] (Inception expects this when using transform_input=False default)
        # Actually torchvision Inception uses ImageNet normalization internally if transform_input=True.
        # For safety, use ImageNet stats explicitly.
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            feats = self.model(x)  # (N, 2048)

        return feats.cpu().numpy()

    def update_real(self, imgs_uint8_rgb):
        """Add a batch of real images. Shape: (N, H, W, 3) or (H, W, 3) uint8."""
        self.real_features.append(self._extract(imgs_uint8_rgb))

    def update_fake(self, imgs_uint8_rgb):
        """Add a batch of fake/generated images. Shape: (N, H, W, 3) or (H, W, 3) uint8."""
        self.fake_features.append(self._extract(imgs_uint8_rgb))

    def compute(self) -> float:
        """Compute FID score. Requires at least 2 samples in each set."""
        from scipy.linalg import sqrtm

        real = np.concatenate(self.real_features, axis=0)
        fake = np.concatenate(self.fake_features, axis=0)

        if real.shape[0] < 2 or fake.shape[0] < 2:
            return float('nan')

        mu_r, mu_f = real.mean(axis=0), fake.mean(axis=0)
        sigma_r = np.cov(real, rowvar=False)
        sigma_f = np.cov(fake, rowvar=False)

        diff = mu_r - mu_f
        covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean))
        return fid

    def reset(self):
        self.real_features = []
        self.fake_features = []


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

_lpips_model = None


def calculate_lpips(img: np.ndarray, img2: np.ndarray, **kwargs) -> float:
    """Calculate LPIPS (AlexNet). Input: uint8 HWC BGR images [0, 255]."""
    import lpips as lpips_lib

    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips_lib.LPIPS(net='alex')

    img_tensor = lpips_lib.im2tensor(img)
    img2_tensor = lpips_lib.im2tensor(img2)
    return _lpips_model(img_tensor, img2_tensor).item()


# ---------------------------------------------------------------------------
# UCIQE
# ---------------------------------------------------------------------------

def calculate_uciqe(img_bgr: np.ndarray, **kwargs) -> float:
    """Calculate UCIQE. Input: uint8 HWC BGR image."""
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    coe_metric = [0.4680, 0.2745, 0.2576]

    img_lum = img_lab[..., 0] / 255.0
    img_a = img_lab[..., 1] / 255.0
    img_b = img_lab[..., 2] / 255.0

    img_chr = np.sqrt(np.square(img_a) + np.square(img_b))
    img_sat = img_chr / np.sqrt(np.square(img_chr) + np.square(img_lum))
    aver_sat = np.mean(img_sat)
    aver_chr = np.mean(img_chr)
    var_chr = np.sqrt(np.mean(abs(1 - np.square(aver_chr / img_chr))))

    nbins = 65536
    hist, bins = np.histogram(img_lum, nbins)
    cdf = np.cumsum(hist) / np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0] - 1) / (nbins - 1), (ihigh[0][0] - 1) / (nbins - 1)]
    con_lum = tol[1] - tol[0]

    return coe_metric[0] * var_chr + coe_metric[1] * con_lum + coe_metric[2] * aver_sat


# ---------------------------------------------------------------------------
# UIQM
# ---------------------------------------------------------------------------

def _uicm(img: np.ndarray) -> float:
    img = img.astype(np.float64)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    RG = R - G
    YB = (R + G) / 2 - B
    K = R.shape[0] * R.shape[1]

    RG1 = np.sort(RG.reshape(1, K))[0]
    alphaL, alphaR = 0.1, 0.1
    RG1 = RG1[int(alphaL * K + 1):int(K * (1 - alphaR))]
    N = K * (1 - alphaR - alphaL)
    meanRG = np.sum(RG1) / N
    deltaRG = np.sqrt(np.sum((RG1 - meanRG) ** 2) / N)

    YB1 = np.sort(YB.reshape(1, K))[0]
    YB1 = YB1[int(alphaL * K + 1):int(K * (1 - alphaR))]
    meanYB = np.sum(YB1) / N
    deltaYB = np.sqrt(np.sum((YB1 - meanYB) ** 2) / N)

    return -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + 0.1586 * np.sqrt(deltaYB ** 2 + deltaRG ** 2)


def _uiconm(img: np.ndarray) -> float:
    from skimage import transform as sk_transform

    img = img.astype(np.float64)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    patchez = 5
    m, n = R.shape[0], R.shape[1]
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        R = sk_transform.resize(R, (x, y))
        G = sk_transform.resize(G, (x, y))
        B = sk_transform.resize(B, (x, y))
    m, n = R.shape[0], R.shape[1]
    k1, k2 = m / patchez, n / patchez

    total = 0.0
    for channel in [R, G, B]:
        amee = 0.0
        for i in range(0, m, patchez):
            for j in range(0, n, patchez):
                im = channel[i:i + patchez, j:j + patchez]
                Max, Min = np.max(im), np.min(im)
                if (Max != 0 or Min != 0) and Max != Min:
                    amee += np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
        total += (1 / (k1 * k2)) * np.abs(amee)
    return total


def _uism(img: np.ndarray) -> float:
    from skimage import transform as sk_transform

    img = img.astype(np.float64)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    SobelR = np.abs(ndimage.convolve(R, hx, mode='nearest') + ndimage.convolve(R, hy, mode='nearest'))
    SobelG = np.abs(ndimage.convolve(G, hx, mode='nearest') + ndimage.convolve(G, hy, mode='nearest'))
    SobelB = np.abs(ndimage.convolve(B, hx, mode='nearest') + ndimage.convolve(B, hy, mode='nearest'))

    patchez = 5
    m, n = R.shape[0], R.shape[1]
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        SobelR = sk_transform.resize(SobelR, (x, y))
        SobelG = sk_transform.resize(SobelG, (x, y))
        SobelB = sk_transform.resize(SobelB, (x, y))

    m, n = SobelR.shape[0], SobelR.shape[1]
    k1, k2 = m / patchez, n / patchez

    lambdas = [0.299, 0.587, 0.114]
    uism = 0.0
    for ch, lam in zip([SobelR, SobelG, SobelB], lambdas):
        eme = 0.0
        for i in range(0, m, patchez):
            for j in range(0, n, patchez):
                im = ch[i:i + patchez, j:j + patchez]
                Max, Min = np.max(im), np.min(im)
                if Max != 0 and Min != 0:
                    eme += np.log(Max / Min)
        uism += lam * (2 / (k1 * k2)) * np.abs(eme)
    return uism


def calculate_uiqm(img_rgb: np.ndarray, **kwargs) -> float:
    """Calculate UIQM. Input: uint8 HWC RGB image."""
    x = img_rgb.astype(np.float32)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return c1 * _uicm(x) + c2 * _uism(x) + c3 * _uiconm(x)
