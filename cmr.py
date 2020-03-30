#.......................................................................
#
# cmr - cleanup messy room
#       using image processing, for teleconference.
#
# 2020. 4. 1 by taroh (sasaki.taroh@gmail.com, facebook.com/taroh.zuo)
#
#.......................................................................
#
import cv2 as cv
import numpy as np
import sys
import argparse
import textwrap

#.......................................................................
def negative(img):
    return 255 - img

def blur(img, blursize):
    return cv.GaussianBlur(img, (blursize, blursize), 0)

def mosaic(img, mosaicsize):
    retimg = np.zeros_like(img)
    for y in range(0, img.shape[0], mosaicsize):
        yy = y + mosaicsize
        if img.shape[0] < yy:
            yy = img.shape[0]
        for x in range(0, img.shape[1], mosaicsize):
            xx = x + mosaicsize
            if img.shape[1] < xx:
                xx = img.shape[1]
            retimg[y:yy, x:xx, 0] = img[y, x, 0]
            retimg[y:yy, x:xx, 1] = img[y, x, 1]
            retimg[y:yy, x:xx, 2] = img[y, x, 2]
    return retimg

def hold():
    while True:
        inkey = cv.waitKey(1) & 0xFF
        if inkey == ord(' '): # until SPACE key hit
            return

def camera_check(shrinkrate):
    cap = cv.VideoCapture(0)
    try:
        ret, inimage = cap.read() # capture 1 frame
    except:
        print("camera error.")
        sys.exit(1)
    cap.release
    return (np.array(inimage.shape[:2]) / shrinkrate).astype(int)

def equalize(img):
    yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv.equalizeHist(yuv[:, :, 0])
    return cv.cvtColor(yuv, cv.COLOR_YUV2BGR)

def autoresize(img, targetsize):
    imgy, imgx = img.shape[:2]
    if targetsize[1] / targetsize[0] < imgx / imgy:
        zm = targetsize[0] / imgy
    else:
        zm = targetsize[1] / imgx
    sy = (targetsize[0] - imgy) / 2
    sx = (targetsize[1] - imgx) / 2
    T = np.float32([[zm, 0., 0.], [0., zm, 0.]])
    img = cv.warpAffine(img, T, tuple(targetsize[::-1]))
    return img

def autocrop(img, targetsize):
    imgy, imgx = img.shape[:2]
    sy = (targetsize[0] - imgy) / 2
    sx = (targetsize[1] - imgx) / 2
    T = np.float32([[1., 0., sx], [0., 1., sy]])
    img = cv.warpAffine(img, T, tuple(targetsize[::-1]))
    return img


face_cascade_path = '/usr/local/opt/opencv/share/'\
                    'opencv4/haarcascades/haarcascade_frontalface_default.xml'
eye_cascade_path = '/usr/local/opt/opencv/share/'\
                   'opencv4/haarcascades/haarcascade_eye.xml'

face_cascade = cv.CascadeClassifier(face_cascade_path)
eye_cascade = cv.CascadeClassifier(eye_cascade_path)

def detectfaces(img, ignoreeyes):
    gimg = cv.equalizeHist(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    faces = face_cascade.detectMultiScale(gimg)
    if len(faces) == 0 or ignoreeyes:
        return faces
    retfaces = []
    for x, y, w, h in faces:
        face = img[y: y + h, x: x + w]
        gface = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gface)
        if len(eyes) == 2:
            retfaces.append([x, y, w, h])
    return retfaces


def circfacemask(masksize, faces, isinner):
    mask = np.zeros(masksize, dtype = np.uint8)
    if len(faces) == 0:
        return mask
    for x, y, w, h in faces:
        cx, cy = int(x + w / 2), int(y + h / 2)
        if isinner:
            r = int(min(w, h) / 2)
        else:
            r = int(np.sqrt(w * w + h * h) / 2)
        cv.circle(mask, (cx, cy), r, 255, -1)
    return mask

def sqfacemask(masksize, faces):
    mask = np.zeros(masksize, dtype = np.uint8)
    if len(faces) == 0:
        return mask
    for x, y, w, h in faces:
        cv.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask

########################################################################
# main
#.......................................................................
# parse args

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description = textwrap.dedent('''\
    cmr - cleanup messy room using image processing.

        prepare OpenCV compatible USB cam, run the script
        and feed for teleconference.

        default action blanks out your messy room, remains inside the outer
        circle of face(s).
        try the --innercircle to cut-off your hair,
                --square if your face is square.

        the --blur [PIXELS], --mosaic [PIXELS], --nagative
        blurs/mosaics/negates your room, in order.

        specify FILENAME for fancy background:
            the --picture {resize|crop} autoresizes/autocrops FILENAME
            to fit your cam.
            the --resize force resizes FILENAME and crops for cam.
            with the --picture chase, FILENAME chases your face
            (to specify center of the FILENAME, try the --center).

        if your face is more messy than your room, try the --reverse
        which swaps foreground (face)/background (room).
    '''),
    epilog = textwrap.dedent('''\
                 key operations:
                     [space]: hold (toggle)
                     q: quit

    '''))
parser.add_argument('FILENAME', nargs = '?', default = None,
                    help = 'use FILENAME for background picture')
parser.add_argument('--shrinkrate', type = float,
                    default = [2.],
                    help = 'camera shrink rate')
parser.add_argument('--picture', '-P',
                    choices = ['resize', 'r', 'crop', 'c', 'chase', 'C'],
                    help = 'resize|r: auto resize FILENAME (default), '\
                           'crop|c: auto crop FILENAME, '\
                           'chase|C: FILENAME chases a face',
                    default = 'resize')
parser.add_argument('--center', '-C', nargs = 2, type = float,
                    default = [.5, .5],
                    help = '(chase mode) specify center (<-0..1->, ^0..1v)')
parser.add_argument('--resize', '-R', nargs = 1, type = float,
                    default = [1.],
                    help = 'force resize FILENAME')
parser.add_argument('--blank', action = 'store_true',
                    help = 'no background (default when no FILENAME)')
parser.add_argument('--blur', nargs = '?',
                    default = 0, const = 35, type = int,
                    help = 'blur background [blur size, odd number]')
parser.add_argument('--mosaic', nargs = '?',
                    default = 0, const = 40, type = int,
                    help = 'mosaic background [mosaic size]')
parser.add_argument('--negative', action = 'store_true',
                    help = 'negate background')
parser.add_argument('--reverse', action = 'store_true',
                    help = 'reverse foreground and background')
parser.add_argument('--noequalize', '-z', action = 'store_true',
                    help = 'do not equalize camera input')
parser.add_argument('--innercircle', action = 'store_true',
                    help = 'crop faces by inner circle '\
                           '(default: outer circle)')
parser.add_argument('--square', action = 'store_true',
                    help = 'crop faces by square')
parser.add_argument('--ignoreeyes', action = 'store_true',
                    help = 'ignore eye detection')
args = parser.parse_args()
if args.FILENAME is None and \
   not 0 < args.blur and 0 < args.mosaic and not args.negative:
        args.blank = True
print(args)

#.......................................................................
# check camera

imsize = camera_check(args.shrinkrate)
print('camera', imsize[1], 'x', imsize[0])

#.......................................................................
# process picture file if given

chase = False
if args.FILENAME is not None:
    fname = args.FILENAME
    bg = cv.imread(fname, cv.IMREAD_COLOR)
    if bg is None:
         print(sys.argv[0], ': cannot open file ', fname, sep = '')
         sys.exit(1)
    if args.picture == 'resize' or args.picture == 'r':
        bg = autoresize(bg, imsize)
    elif args.picture == 'crop' or args.picture == 'c':
        print(bg.shape)
        offx = imsize[1] * .5 - bg.shape[1] * args.resize[0] * args.center[0]
        offy = imsize[0] * .5 - bg.shape[0] * args.resize[0] * args.center[1]
        print(offx, offy)
        T = np.float32(
                [[args.resize[0], 0., offx], [0., args.resize[0], offy]])
        bg = cv.warpAffine(bg, T, tuple(imsize[::-1]))
    elif args.picture == 'chase' or args.picture == 'C':
        bgorg = cv.resize(bg, dsize = None, 
                    fx = args.resize[0], fy = args.resize[0])
        bg = autocrop(bgorg, imsize)
        chase = True
    else: # NEVERREACHED
        sys.exit(99)
    cv.imshow(fname, bg)
    cv.waitKey(1000)
else:
    fname = 'camera'
    bg = np.zeros((imsize[0], imsize[1], 3), dtype = np.uint8)

#.......................................................................
# main loop

cap = cv.VideoCapture(0)
mask = np.zeros((imsize[0], imsize[1]), dtype = np.uint8)
bgmask = 255 - mask
while True:
    ret, camimage = cap.read() # capture 1 frame
    camimage = cv.resize(camimage, dsize = (imsize[1], imsize[0]))
    faces = detectfaces(camimage, args.ignoreeyes)
    if not args.noequalize:
        camimage = equalize(camimage)
    if 0 < len(faces):
        if chase:
            if 0 < len(faces):
                x, y, w, h = faces[0]
            else:
                y, x = imsize[0] / 2, imsize[1] / 2
                w, h = 0, 0
            offx = x + w / 2 - bgorg.shape[1] * args.center[0]
            offy = y + h / 2 - bgorg.shape[0] * args.center[1]
            T = np.float32([[1., 0., offx], [0., 1., offy]])
            bg = cv.warpAffine(bgorg, T, tuple(imsize[::-1]))
        if args.square:
            mask = sqfacemask(camimage.shape[:2], faces)
        else:
            mask = circfacemask(camimage.shape[:2], faces, args.innercircle)
        bgmask = 255 - mask
        if args.reverse:
            mask, bgmask = bgmask, mask
    if 0 < args.mosaic:
        bg = mosaic(camimage, args.mosaic)
    elif 0 < args.blur or args.negative:
        bg = camimage
        if 0 < args.blur:
            bg = blur(bg, int(args.blur / 2) * 2 + 1)
        if args.negative:
            bg = negative(bg)

    outimage = camimage & np.dstack((mask, mask, mask)) |\
               bg & np.dstack((bgmask, bgmask, bgmask))
    cv.imshow(fname, outimage) # then display
    inkey = cv.waitKey(1) & 0xFF
    if inkey == ord('q'): # until SPACE key hit
        break
    elif inkey == ord(' '): # until SPACE key hit
        hold()

cv.destroyWindow('camera')
sys.exit(0)
