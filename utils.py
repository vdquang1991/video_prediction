import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import _reduction as _Reduction
from torch.nn.functional import conv2d
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub

i3d_model = None
# lpips_model = None
_URL = 'http://rail.eecs.berkeley.edu/models/lpips'

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        # m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def normalize_data(sequence):
    # sequence = [B, T, H, W, C]
    sequence.transpose_(0, 1)
    sequence.transpose_(3, 4).transpose_(2, 3)
    # sequence = [T, B, C, H, W]
    return sequence

def kl_criterion(mu1, logvar1, mu2, logvar2, bs):
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
    return kld.sum() / bs

# class KL_criterion(nn.Module):
#     def __init__(self, bs):
#         super(KL_criterion, self).__init__()
#         self.bs = bs
#
#     def call(self, mu1, logvar1, mu2, logvar2):
#         sigma1 = logvar1.mul(0.5).exp()
#         sigma2 = logvar2.mul(0.5).exp()
#         kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
#         return kld.sum() / self.bs


def _fspecial_gaussian(size, channel, sigma):
    coords = torch.tensor([(x - (size - 1.) / 2.) for x in range(size)])
    coords = -coords ** 2 / (2. * sigma ** 2)
    grid = coords.view(1, -1) + coords.view(-1, 1)
    grid = grid.view(1, -1)
    grid = grid.softmax(-1)
    kernel = grid.view(1, 1, size, size)
    kernel = kernel.expand(channel, 1, size, size).contiguous()
    return kernel


def _ssim(input, target, max_val, k1, k2, channel, kernel):
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mu1 = conv2d(input, kernel, groups=channel)
    mu2 = conv2d(target, kernel, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(input * input, kernel, groups=channel) - mu1_sq
    sigma2_sq = conv2d(target * target, kernel, groups=channel) - mu2_sq
    sigma12 = conv2d(input * target, kernel, groups=channel) - mu1_mu2

    v1 = 2 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    return ssim, v1 / v2


def ssim_loss(input, target, max_val, filter_size=11, k1=0.01, k2=0.03,
              sigma=1.5, kernel=None, size_average=None, reduce=None, reduction='mean'):
    r"""ssim_loss(input, target, max_val, filter_size, k1, k2,
                  sigma, kernel=None, size_average=None, reduce=None, reduction='mean') -> Tensor
    Measures the structural similarity index (SSIM) error.
    See :class:`~torch.nn.SSIMLoss` for details.
    """

    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, target.dim(-2), target.dim(-1))
    elif dim == 3:
        input = input.expand(1, input.dim(-3), input.dim(-2), input.dim(-1))
        target = target.expand(1, target.dim(-3), target.dim(-2), target.dim(-1))
    elif dim != 4:
        raise ValueError('Expected 2, 3, or 4 dimensions (got {})'.format(dim))

    _, channel, _, _ = input.size()

    if kernel is None:
        kernel = _fspecial_gaussian(filter_size, channel, sigma)
    kernel = kernel.to(device=input.device)

    ret, _ = _ssim(input, target, max_val, k1, k2, channel, kernel)

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

def _ssim_wrapper(sample, gt):
    """
    Computes the pixel-averaged SSIM between two videos.
    Parameters
    ----------
    sample : torch.*.Tensor
        Tensor representing a video, of shape (length, batch, channels, width, height) and with float values lying in
        [0, 1].
    gt : torch.*.Tensor
        Tensor representing a video, of shape (length, batch, channels, width, height) and with float values lying in
        [0, 1]. Its shape should be the same as sample.
    Returns
    -------
    torch.*.Tensor
        Tensor of pixel-averaged SSIM between the input videos, of shape (length, batch, channels).
    """
    nt, bsz = sample.shape[0], sample.shape[1]
    img_shape = sample.shape[2:]
    ssim = ssim_loss(sample.view(nt * bsz, *img_shape), gt.view(nt * bsz, *img_shape), max_val=1., reduction='none')
    return ssim.mean(dim=[2, 3]).view(nt, bsz, img_shape[0])


def _lpips_wrapper(sample, gt, lpips_model):
    """
    Computes the frame-wise LPIPS between two videos.
    Parameters
    ----------
    sample : torch.*.Tensor
        Tensor representing a video, of shape (length, batch, channels, width, height) and with float values lying in
        [0, 1].
    gt : torch.*.Tensor
        Tensor representing a video, of shape (length, batch, channels, width, height) and with float values lying in
        [0, 1]. Its shape should be the same as sample.
    Returns
    -------
    torch.*.Tensor
        Tensor of frame-wise LPIPS between the input videos, of shape (length, batch).
    """
    nt, bsz = sample.shape[0], sample.shape[1]
    img_shape = sample.shape[2:]
    # Switch to three color channels for grayscale videos
    if img_shape[0] == 1:
        sample_ = sample.repeat(1, 1, 3, 1, 1)
        gt_ = gt.repeat(1, 1, 3, 1, 1)
    else:
        sample_ = sample
        gt_ = gt
    lpips = lpips_model(sample_.view(nt * bsz, 3, *img_shape[1:]), gt_.view(nt * bsz, 3, *img_shape[1:]))
    return lpips.view(nt, bsz)


def cal_lpips(pred_vid, gt_vid, lpips_dist, n_channel=3, image_size=64):
    # Repeat n_channel from 1 to 3
    if n_channel == 1:
        pred_vid = pred_vid.repeat_interleave(3, dim=2)
        gt_vid = gt_vid.repeat_interleave(3, dim=2)

    pred_vid = pred_vid.view(-1, 3, image_size, image_size)     # T*B, C, H , W
    gt_vid = gt_vid.view(-1, 3, image_size, image_size)

    # Norm to [-1, 1]
    pred_vid = pred_vid * 2 - 1
    gt_vid = gt_vid * 2 - 1
    d = lpips_dist(pred_vid, gt_vid)
    return d

def fvd_preprocess(videos, source_resolution, target_resolution=(224, 224), batch_size=2, clip_len=20):
    videos = tf.convert_to_tensor(videos * 255.0, dtype=tf.float32)
    # videos_shape = (batch_size, clip_len, target_resolution[0], target_resolution[1], 3)
    all_frames = tf.reshape(videos, shape=(batch_size * clip_len, source_resolution[0], source_resolution[1], 3))
    resized_videos = tf.image.resize(all_frames, size=target_resolution)
    target_shape = (batch_size, clip_len, target_resolution[0], target_resolution[1], 3)
    output_videos = tf.reshape(resized_videos, target_shape)
    scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
    return scaled_videos

def create_id3_embedding(videos):
    """Get id3 embeddings."""
    global i3d_model
    module_spec = 'https://tfhub.dev/deepmind/i3d-kinetics-400/1'

    if not i3d_model:
        base_model = hub.load(module_spec)
        input_tensor = base_model.graph.get_tensor_by_name('input_frames:0')
        i3d_model = base_model.prune(input_tensor, 'RGB/inception_i3d/Mean:0')

    output = i3d_model(videos)
    return output

def calculate_fvd(real_activations, generated_activations):
    return tfgan.eval.frechet_classifier_distance_from_activations(real_activations, generated_activations)

def fvd(video_1, video_2, image_size, batch_size, clip_len):
    video_1 = fvd_preprocess(video_1, source_resolution=(image_size, image_size), target_resolution=(224, 224),
                             batch_size=batch_size, clip_len=clip_len)
    video_2 = fvd_preprocess(video_2, source_resolution=(image_size, image_size), target_resolution=(224, 224),
                             batch_size=batch_size, clip_len=clip_len)
    x = create_id3_embedding(video_1)
    y = create_id3_embedding(video_2)
    result = calculate_fvd(x, y)
    return result

def gdl_loss(gen_frames, gt_frames, alpha, device):
    """
    Calculates the sum of GDL losses between the predicted and ground truth frames.
    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param alpha: The power to which each gradient term is raised.
    @return: The GDL loss.
    """
    dtype = torch.cuda.FloatTensor if device=="cuda" else torch.FloatTensor
    filter_x_values = np.array(
        [
            [[[-1, 1, 0]], [[0, 0, 0]], [[0, 0, 0]]],
            [[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]],
            [[[0, 0, 0]], [[0, 0, 0]], [[-1, 1, 0]]],
        ],
        dtype=np.float32,
    )
    filter_x = nn.Conv2d(3, 3, (1, 3), padding=(0, 1))
    filter_x.weight = nn.Parameter(torch.from_numpy(filter_x_values))

    filter_y_values = np.array(
        [
            [[[-1], [1], [0]], [[0], [0], [0]], [[0], [0], [0]]],
            [[[0], [0], [0]], [[-1], [1], [0]], [[0], [0], [0]]],
            [[[0], [0], [0]], [[0], [0], [0]], [[-1], [1], [0]]],
        ],
        dtype=np.float32,
    )
    filter_y = nn.Conv2d(3, 3, (3, 1), padding=(1, 0))
    filter_y.weight = nn.Parameter(torch.from_numpy(filter_y_values))

    filter_x = filter_x.type(dtype)
    filter_y = filter_y.type(dtype)

    gen_dx = filter_x(gen_frames)
    gen_dy = filter_y(gen_frames)
    gt_dx = filter_x(gt_frames)
    gt_dy = filter_y(gt_frames)

    grad_diff_x = torch.pow(torch.abs(gt_dx - gen_dx), alpha)
    grad_diff_y = torch.pow(torch.abs(gt_dy - gen_dy), alpha)

    grad_total = torch.stack([grad_diff_x, grad_diff_y])

    return torch.mean(grad_total)


class GDL(nn.Module):
    def __init__(self, alpha=1, temporal_weight=None):
        """
        Args:
            alpha: hyper parameter of GDL loss, float
            temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        self.alpha = alpha
        self.temporal_weight = temporal_weight

    def __call__(self, gt, pred):
        """
        pred --- tensor with shape (B, T, ...)
        gt --- tensor with shape (B, T, ...)
        """
        gt_shape = gt.shape
        if len(gt_shape) == 5:
            B, T, _, _, _ = gt.shape
        elif len(gt_shape) == 6:  # for warped frames, (N, num_future_frames, num_past_frames, C, H, W)
            B, T, TP, _, _, _ = gt.shape
        gt = gt.flatten(0, -4)
        pred = pred.flatten(0, -4)

        gt_i1 = gt[:, :, 1:, :]
        gt_i2 = gt[:, :, :-1, :]
        gt_j1 = gt[:, :, :, :-1]
        gt_j2 = gt[:, :, :, 1:]

        pred_i1 = pred[:, :, 1:, :]
        pred_i2 = pred[:, :, :-1, :]
        pred_j1 = pred[:, :, :, :-1]
        pred_j2 = pred[:, :, :, 1:]

        term1 = torch.abs(gt_i1 - gt_i2)
        term2 = torch.abs(pred_i1 - pred_i2)
        term3 = torch.abs(gt_j1 - gt_j2)
        term4 = torch.abs(pred_j1 - pred_j2)

        if self.alpha != 1:
            gdl1 = torch.pow(torch.abs(term1 - term2), self.alpha)
            gdl2 = torch.pow(torch.abs(term3 - term4), self.alpha)
        else:
            gdl1 = torch.abs(term1 - term2)
            gdl2 = torch.abs(term3 - term4)

        if self.temporal_weight is not None:
            assert self.temporal_weight.shape[0] == T, "Mismatch between temporal_weight and predicted sequence length"
            w = self.temporal_weight.to(gdl1.device)
            _, C, H, W = gdl1.shape
            _, C2, H2, W2 = gdl2.shape
            if len(gt_shape) == 5:
                gdl1 = gdl1.reshape(B, T, C, H, W)
                gdl2 = gdl2.reshape(B, T, C2, H2, W2)
                gdl1 = gdl1 * w[None, :, None, None, None]
                gdl2 = gdl2 * w[None, :, None, None, None]
            elif len(gt_shape) == 6:
                gdl1 = gdl1.reshape(B, T, TP, C, H, W)
                gdl2 = gdl2.reshape(B, T, TP, C2, H2, W2)
                gdl1 = gdl1 * w[None, :, None, None, None, None]
                gdl2 = gdl2 * w[None, :, None, None, None, None]

        # gdl1 = gdl1.sum(-1).sum(-1)
        # gdl2 = gdl2.sum(-1).sum(-1)

        # gdl_loss = torch.mean(gdl1 + gdl2)
        gdl1 = gdl1.mean()
        gdl2 = gdl2.mean()
        gdl_loss = gdl1 + gdl2

        return gdl_loss