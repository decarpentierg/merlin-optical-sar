import numpy as np
import PIL.Image
from IPython.display import HTML, display
from urllib import parse
from io import BytesIO
import base64


def simple_equalization_8bit(im, percentiles=5):
    """
    Simple 8-bit requantization by linear stretching.

    Args:
        im (np.array): image to requantize
        percentiles (int): percentage of the darkest and brightest pixels to saturate

    Returns:
        numpy array with the quantized uint8 image
    """
    mi, ma = np.percentile(im[np.isfinite(im)], (percentiles, 100 - percentiles))
    im = np.clip(im, mi, ma)
    im = (im - mi) / (ma - mi) * 255
    return im.astype(np.uint8)


def register(im1, im2, di, dj):
    """
    Register images with known displacement.

    Parameters
    ----------
    im1, im2: ndarrays of same shape
        images to register

    di, dj: ints
        displacement along the i- and j- directions to apply to the 2nd image to register it on the first
    """
    assert im1.shape == im2.shape
    m, n = im1.shape
    # make di and dj as small as possible in absolute value, using the periodicity of the translation
    if di > m // 2:
        di = di - m
    if dj > n // 2:
        dj = dj - n
    # compute translated and cropped images
    if di > 0:  # im2 must be translated to the right, i.e. im1 to the left
        im1 = im1[di:]
        im2 = im2[:-di]
    elif di < 0:  # im2 must be translated to the left
        im1 = im1[:di]
        im2 = im2[-di:]
    if dj > 0:  # im2 must be translated downwards, i.e. im1 upwards
        im1 = im1[:, dj:]
        im2 = im2[:, :-dj]
    elif dj < 0:  # im2 must be translated upwards
        im1 = im1[:, :dj]
        im2 = im2[:, -dj:]
    # return result
    return im1, im2


gallery_style_base = """
    <style>
        .gallery2 {
            position: relative;
            width: auto;
            height: 650px; }
        .gallery2 .index {
            padding: 0;
            margin: 0;
            width: 10.5em;
            list-style: none; }
        .gallery2 .index li {
            margin: 0;
            padding: 0;
            float: left;}
        .gallery2 .index a { /* gallery2 item title */
            display: block;
            background-color: #EEEEEE;
            border: 1px solid #FFFFFF;
            text-decoration: none;
            width: 1.9em;
            padding: 6px; }
        .gallery2 .index a span { /* gallery2 item content */
            display: block;
            position: absolute;
            left: -9999px; /* hidden */
            top: 0em;
            padding-left: 0em; }
        .gallery2 .index a span img{ /* gallery2 item content */
            height: 550px;
            }
        .gallery2 .index li:first-child a span {
            top: 0em;
            left: 10.5em;
            z-index: 99; }
        .gallery2 .index a:hover {
            border: 1px solid #888888; }
        .gallery2 .index a:hover span {
            left: 10.5em;
            z-index: 100; }
    </style>
"""

svg_overlay_style = """
    <style>
        .svg-overlay {
            position: relative;
            display: inline-block;
        }

        .svg-overlay svg {
            position: absolute;
            top: 0;
            left: 0;
        }
    </style>
"""


def urlencoded_jpeg_img(a):
    """
    returns the string of an html img tag with the urlencoded jpeg of 'a'
    supports monochrome (shape = (N,M,1) or (N,M))
    and color arrays (N,M,3)
    """
    fmt = "jpeg"

    # handle color images (3,N,M) -> (N,M,3)
    a = a.squeeze()
    if len(a.shape) == 3 and a.shape[0] == 3:
        a = a.transpose(1, 2, 0)

    f = BytesIO()
    PIL.Image.fromarray(np.uint8(a).squeeze()).save(f, fmt)
    x = base64.b64encode(f.getvalue())
    return """<img src="data:image/jpeg;base64,{}&#10;"/>""".format(x.decode())
    # display using IPython.display.HTML(retval)


def display_gallery(image_urls, image_labels=None, svg_overlays=None):
    """
    image_urls can be a list of urls
    or a list of numpy arrays
    image_labels is a list of strings
    """

    gallery_template = """
    <div class="gallery2">
        <ul class="index">
            {}
        </ul>
    </div>
    """

    li_template = """<li><a href="#">{}<span style="background-color: white;  " ><img src="{}" /></br>{}</span></a></li>"""
    li_template_encoded = """<li><a href="#">{}<span style="background-color: white;  " >{}</br>{}</span></a></li>"""

    li = ""
    idx = 0
    for u in image_urls:
        if image_labels:
            label = image_labels[idx]
        else:
            label = str(idx)

        if svg_overlays:
            svg = svg_overlays[idx]
        else:
            svg = None

        if type(u) == str and parse.urlparse(u).scheme in ("http", "https", "ftp"):  # full url
            li = li + li_template.format(idx, u, label)
        elif type(u) == str:  # assume relative url path
            img = np.asarray(PIL.Image.open(u))
            li = li + li_template_encoded.format(idx, urlencoded_jpeg_img(img), label)
        elif type(u) == np.ndarray:  # input is np.array
            h, w = u.shape[0], u.shape[1]
            div = f'<div class="svg-overlay">{urlencoded_jpeg_img(u)}<svg viewBox="0 0 {w} {h}">{svg}</svg></div>'
            li = li + li_template_encoded.format(idx, div, label)

        idx = idx + 1

    source = gallery_template.format(li)

    display(HTML(source))
    display(HTML(gallery_style_base))
    display(HTML(svg_overlay_style))
