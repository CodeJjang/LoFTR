import glob

from src.loftr import LoFTR, default_cfg
import os
from copy import deepcopy
import torch.nn.functional as F

import torchvision.transforms as transforms

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

from src.networks.MultiscaleTransformerEncoder import MultiscaleTransformerEncoder
from src.utils.plotting import make_matching_figure

def load_loftr():
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()
    return matcher

def load_mt_model():
    file_name = glob.glob('weights/multiscale_transformer.pth')[0]

    checkpoint = torch.load(file_name)

    model = MultiscaleTransformerEncoder(dropout=0.5)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.eval()

    model = model.cuda()
    return model

def predict_loftr(loftr, img0, img1, topk=1000000):
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        loftr(batch)
        mconf = batch['mconf'].cpu().numpy()
        topk_indices = mconf.argsort()[-topk:][::-1]

        mkpts0 = batch['mkpts0_f'].cpu().numpy()[topk_indices]
        mkpts1 = batch['mkpts1_f'].cpu().numpy()[topk_indices]

    return mkpts0, mkpts1

def visualise_keypoints(im1, im2, pts, conf1=None, conf2=None, vis_inliners=True, vis_outliers=True):
    new_cmap = rand_cmap(100, type='bright', first_color_black=True, last_color_black=False, verbose=False)

    pts = [pts[0], pts[1]]

    def draw_keypoints(im, pts, imW=im1.shape[0]):
        ptsA, ptsB = pts
        draw = ImageDraw.Draw(im)
        r = 4
        for j in range(0, ptsA.shape[0]):
            R, G, B, A = new_cmap(j)
            is_inline = np.linalg.norm(ptsA[j] - ptsB[j]) < 5
            if is_inline and vis_inliners:
                draw.ellipse([ptsA[j, 0] - r, ptsA[j, 1] - r, ptsA[j, 0] + r, ptsA[j, 1] + r], fill=(255, 0, 0, 255))
                draw.ellipse([ptsB[j, 0] - r + imW, ptsB[j, 1] - r, ptsB[j, 0] + r + imW, ptsB[j, 1] + r],
                             fill=(255, 0, 0, 255))
                R, G, B, A = 0, 255, 0, 1
                draw.line([ptsA[j, 0], ptsA[j, 1], ptsB[j, 0] + imW, ptsB[j, 1]], width=r, fill=(int(R * 255),
                                                                                                 int(G * 255),
                                                                                                 int(B * 255),
                                                                                                 int(A * 255)))
            elif not is_inline and vis_outliers:
                draw.ellipse([ptsA[j, 0] - r, ptsA[j, 1] - r, ptsA[j, 0] + r, ptsA[j, 1] + r], fill=(255, 0, 0, 255))
                draw.ellipse([ptsB[j, 0] - r + imW, ptsB[j, 1] - r, ptsB[j, 0] + r + imW, ptsB[j, 1] + r],
                             fill=(255, 0, 0, 255))
                R, G, B, A = 255, 0, 0, 1
                draw.line([ptsA[j, 0], ptsA[j, 1], ptsB[j, 0] + imW, ptsB[j, 1]], width=r, fill=(int(R * 255),
                                                                                                 int(G * 255),
                                                                                                 int(B * 255),
                                                                                                 int(A * 255)))
            elif not vis_inliners and not vis_outliers:
                draw.ellipse([ptsA[j, 0] - r, ptsA[j, 1] - r, ptsA[j, 0] + r, ptsA[j, 1] + r], fill=(255, 0, 0, 255))
                draw.ellipse([ptsB[j, 0] - r + imW, ptsB[j, 1] - r, ptsB[j, 0] + r + imW, ptsB[j, 1] + r],
                             fill=(255, 0, 0, 255))
                draw.line([ptsA[j, 0], ptsA[j, 1], ptsB[j, 0] + imW, ptsB[j, 1]], width=r, fill=(int(R * 255),
                                                                                                 int(G * 255),
                                                                                                 int(B * 255),
                                                                                                 int(A * 255)))

    fig = plt.figure(figsize=(30, 15))
    axes = fig.subplots(nrows=1, ncols=1)

    # Pad to make the heights the same
    if im1.shape[1] < im2.shape[1]:
        W_n, H_n = im1.shape
        W_2, H_2 = im2.shape
        A = H_2 / H_n

        H_n = H_2
        W_n = int(A * W_n)

        im1 = np.resize(im1, (W_n, H_n))
        pts[0] = pts[0] * A
    elif im1.shape[1] > im2.shape[1]:
        W_1, H_1 = im1.shape
        W_n, H_n = im2.shape
        A = H_1 / H_n

        H_n = H_1
        W_n = int(A * W_n)
        im2 = np.resize(im2, (W_n, H_n))
        pts[1] = pts[1] * A

    im1 = np.array(im1)
    im2 = np.array(im2)

    im = Image.fromarray(np.hstack((im1, im2)))
    draw_keypoints(im, pts, imW=im1.shape[1])
    axes.imshow(im)

    plt.axis('off')
    return im

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

def best_matches(sim, cond1=None, cond2=None, topk=8000, T=0.3, nn=1):
    ''' Find the best matches for a given NxN matrix.
        Optionally, pass in the actual indices corresponding to this matrix
        (cond1, cond2) and update the matches according to these indices.
    '''
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]

    ids1 = torch.arange(0, sim.shape[0]).to(nn12.device)
    mask = (ids1 == nn21[nn12])

    matches = torch.stack([ids1[mask], nn12[mask]])

    preds = sim[ids1[mask], nn12[mask]]
    res, ids = preds.sort()
    ids = ids[res > T]

    if not (cond1 is None) and not (cond2 is None):
        cond_ids1 = cond1[ids1[mask]]
        cond_ids2 = cond2[nn12[mask]]

        matches = torch.stack([cond_ids1, cond_ids2])

    top_matches = matches[:, ids[-topk:]]
    return top_matches.t(), None

def extract_descriptors_around_keypoints(desc_model, im1_orig, im2_orig, kps1, kps2):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    im1, im2 = transform(im1_orig), transform(im2_orig)
    _, im1_h, im1_w = im1.shape
    _, im2_h, im2_w = im2.shape
    patch_h = patch_w = 64
    im1 = F.pad(im1, (patch_h // 2, patch_h // 2, patch_h // 2, patch_h // 2))
    im2 = F.pad(im2, (patch_h // 2, patch_h // 2, patch_h // 2, patch_h // 2))
    patches = np.zeros((len(kps1), 2, patch_h, patch_w))
    for i, (kp1, kp2) in enumerate(zip(kps1, kps2)):
        kp1, kp2 = kp1 + patch_h // 2, kp2 + patch_h // 2
        for j, (kp, im) in enumerate(zip([kp1, kp2], [im1, im2])):
            min_x = int(kp[0]) - patch_h // 2
            max_x = int(kp[0]) + patch_h // 2
            min_y = int(kp[1]) - patch_h // 2
            max_y = int(kp[1]) + patch_h // 2
            patch = im.squeeze()[min_y: max_y, min_x: max_x]
            patches[i, j] = patch

    patches = torch.from_numpy(patches)
    desc12 = torch.zeros((len(kps1), 128), dtype=torch.float32)
    desc21 = torch.zeros((len(kps2), 128), dtype=torch.float32)
    step_size = 1000
    im1_patches = patches[:, 0].unsqueeze(1).float()
    im2_patches = patches[:, 1].unsqueeze(1).float()
    with torch.no_grad():
        for i in range(0, len(im1_patches.squeeze()), step_size):
            embs = desc_model(im1_patches[i: i+step_size].cuda(), im2_patches[i: i+step_size].cuda())
            desc12[i: i+step_size] = embs['Emb1'].detach().cpu()
            desc21[i: i+step_size] = embs['Emb2'].detach().cpu()
    sim_matrix = torch.mm(desc12, desc21.transpose(0, 1))
    return sim_matrix

def main():
    loftr = load_loftr()
    mt_model = load_mt_model()
    # Load example images
    # img0_pth = "assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
    # img1_pth = "assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
    img0_pth = 'D:/multisensor/datasets/Vis-Nir/data/water/0003_rgb.tiff'
    img1_pth = 'D:/multisensor/datasets/Vis-Nir/data/water/0003_nir.tiff'
    img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//8*8, img0_raw.shape[0]//8*8))  # input size shuold be divisible by 8
    img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    mkpts0, mkpts1 = predict_loftr(loftr, img0, img1, topk=200)


    dist = np.linalg.norm(mkpts0 - mkpts1, axis=1)
    inliners = dist[dist < 5]
    outliers = dist[dist >= 5]
    mean_error = 0
    if outliers.any():
        mean_error = np.mean([d for d in outliers])
    print(f'Inliners/Outliers: {len(inliners)}/{len(outliers)}')
    print(f'Mean error: {mean_error}')

    img0_raw = cv2.imread(img0_pth)
    img1_raw = cv2.imread(img1_pth)
    img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//8*8, img0_raw.shape[0]//8*8))  # input size shuold be divisible by 8
    img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))
    visualise_keypoints(img0_raw, img1_raw, (mkpts0, mkpts1))
    plt.show()

if __name__ == '__main__':
    main()