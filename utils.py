import argparse
import torch
import numpy as np
import pylab as plt
from PIL import Image

from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
# =====================================
# helpers
import cv2, glob, re
import transformer_net
import dabnet
import torch
import tqdm
import argparse
import pandas as pd


def build_model(model_name, num_classes):
    if model_name == 'DABNet':
        return dabnet.DABNet(classes=num_classes)

def train_epoch(model, train_loader):
    model.train()
    loss_sum = 0.

    n_batches = len(train_loader)
    pbar = tqdm.tqdm(total=n_batches)
    for i, batch in enumerate(train_loader):
        loss_sum += float(model.train_step(batch))

        pbar.set_description("Training loss: %.4f" % (loss_sum / (i + 1)))
        pbar.update(1)

    pbar.close()
    loss = loss_sum / n_batches

    return {"train_loss": loss}

@torch.no_grad()
def val_epoch(model, val_loader):
    model.eval()
    ae = 0.
    n_samples = 0.

    n_batches = len(val_loader)
    pbar = tqdm.tqdm(total=n_batches)
    for i, batch in enumerate(val_loader):
        pred_count = model.predict(batch, method="counts")

        ae = abs(batch["counts"].cpu().numpy().ravel() - pred_count.ravel()).sum()
        n_samples += batch["counts"].shape[0]

        pbar.set_description("Val mae: %.4f" % (ae / n_samples))
        pbar.update(1)

    pbar.close()
    score = ae / n_samples

    return {"val_score": score, "val_mae":score}

def create_seg_model():
    model = build_model("DABNet", num_classes=19)
    model.eval()
    model = model.cuda()
    model.load_state_dict(torch.load("pretrained/DABNet_cityscapes.pth")["model"])

    return model
@torch.no_grad()
def get_styled_image(style_model, image):
    styled = style_model(image)

    # make same size
    styled = F.interpolate(styled, image.size()[2:], mode='bilinear', align_corners=False)

    img = styled[0].cpu().detach().clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    return img


def create_style_model(style_number=0):
    model_file = glob.glob('styles/*.pth')[style_number]
    transformer = transformer_net.TransformerNet()
    # load model
    state_dict = torch.load(model_file)
    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    transformer.load_state_dict(state_dict)
    transformer.to(0)
    transformer.eval()
    return transformer

def load_image_style(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(img)
    content_image = content_image.unsqueeze(0).to(0)

    return content_image

def load_image(fname):
    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    image = np.asarray(image, np.float32)
    size = image.shape
    mean = np.array([72.3924  , 82.90902 , 73.158325])
    image -= mean
    # image = image.astype(np.float32) / 255.0
    image = image[:, :, ::-1]  # change to RGB
    image = image.transpose((2, 0, 1))  # HWC -> CHW

    return image[None]


def denorm(image):
    mean = np.array([72.3924, 82.90902, 73.158325])
    image_original = (f2l(image[0])[:, :, ::-1] + mean)[:, :, ::-1]
    return image_original

@torch.no_grad()
def get_semseg_image(model, image):
    with torch.no_grad():
        input_var = Variable(torch.FloatTensor(image.copy())).cuda()

    output = model(input_var)
    torch.cuda.synchronize()
    output = output.cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)
    output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

    return output

@torch.no_grad()
def get_masked_image(model, image, category, bg=0):
    with torch.no_grad():
        input_var = Variable(torch.FloatTensor(image.copy())).cuda()

    output = model(input_var)
    torch.cuda.synchronize()
    output = output.cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)
    output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

    bin_mask = ((output == category)).astype('uint8')
    if bg:
        bin_mask = 1 - bin_mask

    image_original = denorm(image)
    masked = bin_mask[:, :, None] * image_original
    # image = Image.fromarray(masked.astype('uint8'))

    return masked

def save_image(fname, image):
    image = Image.fromarray(image.astype('uint8'))
    image.save(fname)

def apply_style():
    pass

def load_weight(cfg):
    # catalog lookup

    f = cfg.MODEL.WEIGHT
    if f.startswith("catalog://"):
        paths_catalog = import_file("maskrcnn_benchmark.config.paths_catalog",
                                    cfg.PATHS_CATALOG, True)
        catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://"):])
        # self.logger.info("{} points to {}".format(f, catalog_f))
        f = catalog_f
    # download url files
    if f.startswith("http"):
        # if the file is a url path, download it and cache it
        cached_f = cache_url(f)
        # self.logger.info("url {} cached in {}".format(f, cached_f))
        f = cached_f
    # convert Caffe2 checkpoint from pkl
    if f.endswith(".pkl"):
        return load_c2_format(cfg, f)

def get_transform(transform):
  if transform == "bgr_normalize":
    PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
    PIXEL_STD = [1., 1., 1.]

    normalize_transform = transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    return transforms.Compose(
      [transforms.ToTensor(),
       BGR_Transform(), normalize_transform])

class BGR_Transform(object):
  def __init__(self):
    pass

  def __call__(self, x):
    return (x * 255)[[2, 1, 0]]

def _denorm(image, mu, var, bgr2rgb=False):
    if image.ndim == 3:
        result = image * var[:, None, None] + mu[:, None, None]
        if bgr2rgb:
            result = result[::-1]
    else:
        result = image * var[None, :, None, None] + mu[None, :, None, None]
        if bgr2rgb:
            result = result[:, ::-1]
    return result


def denormalize(img, mode=0):
    # _img = t2n(img)
    # _img = _img.copy()
    image = t2n(img).copy().astype("float")

    if mode in [1, "rgb"]:
        mu = np.array([0.485, 0.456, 0.406])
        var = np.array([0.229, 0.224, 0.225])
        # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image = _denorm(image, mu, var)

    elif mode in [2, "bgr"]:
        mu = np.array([102.9801, 115.9465, 122.7717])
        var = np.array([1, 1, 1])
        image = _denorm(image, mu, var, bgr2rgb=True).clip(0, 255).round()

    elif mode in [3, "basic"]:
        mu = np.array([0.5, 0.5, 0.5])
        var = np.array([0.5, 0.5, 0.5])

        image = _denorm(image, mu, var)
    # else:

    #     mu = np.array([0.,0.,0.])
    #     var = np.array([1,1,1])

    #     image = _denorm(image, mu, var)
    # print(image)
    return image
def ensure_image_list(T, SIZE_DIVISIBILITY=1):
  if isinstance(T, torch.Tensor):
    image_list = to_image_list([t for t in T],
                               size_divisible=SIZE_DIVISIBILITY)
  else:
    image_list = T

  return image_list

def segm2annList(segm,
                 boxes,
                 scoreList,
                 categoryList,
                 H,
                 W,
                 image_id,
                 mode="yxyx",
                 mask=None,
                 score_threshold=None):

  if len(boxes) == 0:
    return []
  if boxes.max() < 1:
    boxes_denorm = bbox_yxyx_denormalize(boxes, (1, 3, H, W))
  else:
    boxes_denorm = boxes

  if mode == "yxyx":
    boxes_xywh = t2n(yxyx2xywh(boxes_denorm))
  else:
    boxes_xywh = t2n(xyxy2xywh(boxes_denorm))

  annList = []

  for i in range(len(boxes_xywh)):
    score = float(scoreList[i])
    if score_threshold is not None and score < score_threshold:
      continue
    ann = {
      "segmentation": segm[i],
      "bbox": list(map(int, boxes_xywh[i])),
      "image_id": image_id,
      "category_id": int(categoryList[i]),
      "height": H,
      "width": W,
      "score": score
    }

    annList += [ann]

  return annList

def xyxy2xywh(boxes_xyxy):
  x1, y1, x2, y2 = torch.chunk(boxes_xyxy, chunks=4, dim=1)
  h = y2 - y1
  w = x2 - x1

  return torch.cat([x1, y1, w, h], dim=1)

def yxyx2xywh(boxes_yxyx):
  y1, x1, y2, x2 = torch.chunk(boxes_yxyx, chunks=4, dim=1)
  h = y2 - y1
  w = x2 - x1

  return torch.cat([x1, y1, w, h], dim=1)

def bbox_yxyx_denormalize(bbox, image_shape, window_box=None, clamp=True):
  _, _, H, W = image_shape

  if window_box is None:
    window_box = [0, 0, H, W]
  else:
    window_box = list(map(int, window_box))

  H_window_box = window_box[2] - window_box[0]
  W_window_box = window_box[3] - window_box[1]

  scales = torch.FloatTensor(
    [H_window_box, W_window_box, H_window_box, W_window_box])

  shift = torch.FloatTensor(
    [window_box[0], window_box[1], window_box[0], window_box[1]])
  # Translate bounding boxes to image domain
  bbox = bbox * scales + shift
  if clamp:
    bbox = clamp_boxes_yxyx(bbox, image_shape)

  return bbox

def clamp_boxes_yxyx(boxes, image_shape):

  _, _, H, W = image_shape
  # Height
  boxes[:, 0] = boxes[:, 0].clamp(0, H - 1)
  boxes[:, 2] = boxes[:, 2].clamp(0, H - 1)

  # Width
  boxes[:, 1] = boxes[:, 1].clamp(0, W - 1)
  boxes[:, 3] = boxes[:, 3].clamp(0, W - 1)

  return boxes

def get_image(image, annList, dpi=100, **options):
    image = f2l(image).squeeze().clip(0, 255)
    if image.max() > 1:
        image /= 255.

    # box_alpha = 0.5
    # print(image.clip(0, 255).max())
    color_list = colormap(rgb=True) / 255.

    # fig = Figure()
    fig = plt.figure(frameon=False)
    canvas = FigureCanvas(fig)
    fig.set_size_inches(image.shape[1] / dpi, image.shape[0] / dpi)
    # ax = fig.gca()

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    # im = im.clip(0, 1)
    # print(image)
    ax.imshow(image)


    mask_color_id = 0
    for i in range(len(annList)):
        ann = annList[i]

        if "bbox" in ann:
            bbox = ann["bbox"]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2],
                              bbox[3],
                              fill=False,
                              edgecolor='r',
                              linewidth=3.0,
                              alpha=0.5))

        # if show_class:
        if options.get("show_text") == True or options.get("show_text") is None:
            score = ann["score"] or -1
            ax.text(
                bbox[0], bbox[1] - 2,
                "%.1f" % score,
                fontsize=14,
                family='serif',
                bbox=dict(facecolor='g', alpha=1.0, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if "segmentation" in ann:
            mask = ann2mask(ann)["mask"]
            img = np.ones(image.shape)
            # category_id = ann["category_id"]
            # mask_color_id = category_id - 1
            # color_list = ["r", "g", "b","y", "w","orange","purple"]
            # color_mask = color_list[mask_color_id % len(color_list)]
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            # print("color id: %d - category_id: %d - color mask: %s"
                        # %(mask_color_id, category_id, str(color_mask)))
            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = mask

            contour, hier = cv2.findContours(e.copy(),
                                    cv2.RETR_CCOMP,
                                    cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True,
                    facecolor=color_mask,
                    edgecolor="white",
                    linewidth=1.5,
                    alpha=0.7
                    )
                ax.add_patch(polygon)

    canvas.draw()  # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

    fig_image = np.fromstring(
        canvas.tostring_rgb(), dtype='uint8').reshape(
            int(height), int(width), 3)
    plt.close()
    # print(fig_image)
    return fig_image

def ann2mask(ann):
    if "mask" in ann:
        mask = ann["mask"]
    elif "segmentation" in ann:
        # TODO: fix this monkey patch
        if isinstance(ann["segmentation"]["counts"], list):
            ann["segmentation"]["counts"] = ann["segmentation"]["counts"][0]
        mask = mask_utils.decode(ann["segmentation"])
    else:
        x,y,w,h = ann["bbox"]
        img_h, img_w = ann["height"], ann["width"]
        mask = np.zeros((img_h, img_w))
        mask[y:y+h, x:x+w] = 1
    # mask[mask==1] = ann["category_id"]
    return {"mask": mask}

def colormap(rgb=False):
    color_list = np.array([
        0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
        0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
        0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
        1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
        0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
        0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
        0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
        1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
        0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
        0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
        0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
        0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
        0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
        0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
        1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
        1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
        0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
        0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000,
        0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
        0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000,
        0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
        0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286,
        0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714,
        0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000
    ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list

def f2l(X):
    if X.ndim == 3 and (X.shape[2] == 3 or X.shape[2] == 1):
        return X
    if X.ndim == 4 and (X.shape[3] == 3 or X.shape[3] == 1):
        return X

    # CHANNELS FIRST
    if X.ndim == 3:
        return np.transpose(X, (1, 2, 0))
    if X.ndim == 4:
        return np.transpose(X, (0, 2, 3, 1))

    return X

def t2n(x):
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, torch.autograd.Variable):
        x = x.cpu().data.numpy()

    if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.IntTensor,
                      torch.cuda.LongTensor, torch.cuda.DoubleTensor)):
        x = x.cpu().numpy()

    if isinstance(x, (torch.FloatTensor, torch.IntTensor, torch.LongTensor,
                      torch.DoubleTensor)):
        x = x.numpy()

    return x

import os


def imsave(fname, arr, size=None):
    from PIL import Image
    arr = f2l(t2n(arr)).squeeze()
    create_dirs(fname + "tmp")
    #print(arr.shape)
    image = Image.fromarray(arr)

    image.save(fname)

def create_dirs(fname):
    if "/" not in fname:
        return

    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))


from PIL import Image
import torch
import numpy as np

cityscapes_palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                      220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0,
                      70,
                      0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

camvid_palette = [128, 128, 128, 128, 0, 0, 192, 192, 128, 128, 64, 128, 60, 40, 222, 128, 128, 0, 192, 128, 128, 64,
                  64,
                  128, 64, 0, 128, 64, 64, 0, 0, 128, 192]

zero_pad = 256 * 3 - len(cityscapes_palette)
for i in range(zero_pad):
    cityscapes_palette.append(0)


# zero_pad = 256 * 3 - len(camvid_palette)
# for i in range(zero_pad):
#     camvid_palette.append(0)

def cityscapes_colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(cityscapes_palette)

    return new_mask


def camvid_colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(camvid_palette)

    return new_mask


class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = voc_color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image


def voc_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap
