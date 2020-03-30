# cmr
cmr - cleanup messy room using image processing, for teleconference.

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
