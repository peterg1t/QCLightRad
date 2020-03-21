import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.feature import blob_log




def point_detect(imcirclist, minSigma, maxSigma, numSigma, thres):
    k = 0
    detCenterXRegion = []
    detCenterYRegion = []

    print("Finding bibs in phantom...")
    for img in tqdm(imcirclist):
        grey_img = np.array(img, dtype=np.uint8)  # converting the image to grayscale
        blobs_log = blob_log(
            grey_img, min_sigma=minSigma, max_sigma=maxSigma, num_sigma=numSigma, threshold=thres
        )

        centerXRegion = []
        centerYRegion = []
        centerRRegion = []
        grey_ampRegion = []
        for blob in blobs_log:
            y, x, r = blob
            # center = (int(x), int(y))
            centerXRegion.append(x)
            centerYRegion.append(y)
            centerRRegion.append(r)
            grey_ampRegion.append(grey_img[int(y), int(x)])
            # radius = int(r)
            # print('center=', center, 'radius=', radius, 'value=', img[center], grey_img[center])

        xindx = int(centerXRegion[np.argmin(grey_ampRegion)])
        yindx = int(centerYRegion[np.argmin(grey_ampRegion)])
        # rindx = int(centerRRegion[np.argmin(grey_ampRegion)])

        detCenterXRegion.append(xindx)
        detCenterYRegion.append(yindx)

        k = k + 1

    return detCenterXRegion, detCenterYRegion








def roi_sel_FC2(ArrayDicom,ioption,dx,dy):
     # creating the figure extent list for the bib images
    list_extent = []
    print('something here')
    if ioption.startswith(('y', 'yeah', 'yes')):
        ROI1 = {"edge_top": 70, "edge_bottom": 130, "edge_left": 270, "edge_right": 350}
        ROI2 = {"edge_top": 70, "edge_bottom": 130, "edge_left": 680, "edge_right": 760}
        ROI3 = {"edge_top": 150,"edge_bottom": 210,"edge_left": 760,"edge_right": 830}
        ROI4 = {"edge_top": 560,"edge_bottom": 620,"edge_left": 760,"edge_right": 830}
    else:
        ROI1 = {
                "edge_top": 66,
                "edge_bottom": 166,
                "edge_left": 195,
                "edge_right": 295,
            }
        ROI2 = {
               "edge_top": 66,
               "edge_bottom": 166,
               "edge_left": 698,
               "edge_right": 798,
           }
        ROI3 = {
               "edge_top": 582,
               "edge_bottom": 682,
               "edge_left": 698,
               "edge_right": 798,
           }
        ROI4 = {
               "edge_top": 582,
               "edge_bottom": 682,
               "edge_left": 195,
               "edge_right": 295,
           }

   # images for object detection
    imcirclist = []
    imcirc1 = Image.fromarray(
       255
       * ArrayDicom[
           ROI1["edge_top"] : ROI1["edge_bottom"],
           ROI1["edge_left"] : ROI1["edge_right"],
       ]
    )
    imcirc1 = imcirc1.resize((imcirc1.width * 10, imcirc1.height * 10), Image.LANCZOS)

    list_extent.append(
       (
           (ROI1["edge_left"] * dx / 10),
           (ROI1["edge_right"] * dx / 10),
           (ROI1["edge_bottom"] * dy / 10),
           (ROI1["edge_top"] * dy / 10),
       )
   )

    imcirc2 = Image.fromarray(
       255
       * ArrayDicom[
           ROI2["edge_top"] : ROI2["edge_bottom"],
           ROI2["edge_left"] : ROI2["edge_right"],
       ]
   )
    imcirc2 = imcirc2.resize((imcirc2.width * 10, imcirc2.height * 10), Image.LANCZOS)

    list_extent.append(
       (
           (ROI2["edge_left"] * dx / 10),
           (ROI2["edge_right"] * dx / 10),
           (ROI2["edge_bottom"] * dy / 10),
           (ROI2["edge_top"] * dy / 10),
       )
   )

    imcirc3 = Image.fromarray(
       255
       * ArrayDicom[
           ROI3["edge_top"] : ROI3["edge_bottom"],
           ROI3["edge_left"] : ROI3["edge_right"],
       ]
   )
    imcirc3 = imcirc3.resize((imcirc3.width * 10, imcirc3.height * 10), Image.LANCZOS)
    list_extent.append(
       (
           (ROI3["edge_left"] * dx / 10),
           (ROI3["edge_right"] * dx / 10),
           (ROI3["edge_bottom"] * dy / 10),
           (ROI3["edge_top"] * dy / 10),
       )
   )

    imcirc4 = Image.fromarray(
       255
       * ArrayDicom[
           ROI4["edge_top"] : ROI4["edge_bottom"],
           ROI4["edget_left"] : ROI4["edge_right"],
       ]
   )
    imcirc4 = imcirc4.resize((imcirc4.width * 10, imcirc4.height * 10), Image.LANCZOS)

    list_extent.append(
       (
           (ROI4["edge_left"] * dx / 10),
           (ROI4["edge_right"] * dx / 10),
           (ROI4["edge_bottom"] * dy / 10),
           (ROI4["edge_top"] * dy / 10),
       )
   )


    imcirclist.append(imcirc1)
    imcirclist.append(imcirc2)
    imcirclist.append(imcirc3)
    imcirclist.append(imcirc4)

    minSigma=15
    maxSigma=40
    numSigma=10
    thres=0.05
    xdet, ydet = point_detect(imcirclist, minSigma, maxSigma, numSigma, thres)

    profiles = []
    profile1 = np.array(imcirc1, dtype=np.uint8)[:, xdet[0]] / 255
    profile2 = np.array(imcirc2, dtype=np.uint8)[:, xdet[1]] / 255
    profile3 = np.array(imcirc3, dtype=np.uint8)[ydet[2], :] / 255
    profile4 = np.array(imcirc4, dtype=np.uint8)[ydet[3], :] / 255

    profiles.append(profile1)
    profiles.append(profile2)
    profiles.append(profile3)
    profiles.append(profile4)


    return profiles, imcirclist, xdet, ydet, list_extent















def roi_sel_IsoAlign(ArrayDicom,ioption,dx,dy):
     # creating the figure extent list for the bib images
    list_extent = []
    if ioption.startswith(('y', 'yeah', 'yes')):
        ROI1 = {"edge_top": 70, "edge_bottom": 130, "edge_left": 270, "edge_right": 350}
        ROI2 = {"edge_top": 70, "edge_bottom": 130, "edge_left": 680, "edge_right": 760}
        ROI3 = {
            "edge_top": 150,
            "edge_bottom": 210,
            "edge_left": 760,
            "edge_right": 830,
        }
        ROI4 = {
            "edge_top": 560,
            "edge_bottom": 620,
            "edge_left": 760,
            "edge_right": 830,
        }
        ROI5 = {
            "edge_top": 640,
            "edge_bottom": 700,
            "edge_left": 680,
            "edge_right": 760,
        }
        ROI6 = {
            "edge_top": 640,
            "edge_bottom": 700,
            "edge_left": 270,
            "edge_right": 350,
        }
        ROI7 = {
            "edge_top": 560,
            "edge_bottom": 620,
            "edge_left": 200,
            "edge_right": 270,
        }
        ROI8 = {
            "edge_top": 150,
            "edge_bottom": 210,
            "edge_left": 200,
            "edge_right": 270,
        }
    else:
        ROI1 = {
            "edge_top": 280,
            "edge_bottom": 360,
            "edge_left": 360,
            "edge_right": 440,
        }
        ROI2 = {
            "edge_top": 280,
            "edge_bottom": 360,
            "edge_left": 830,
            "edge_right": 910,
        }
        ROI3 = {
            "edge_top": 360,
            "edge_bottom": 440,
            "edge_left": 940,
            "edge_right": 1020,
        }
        ROI4 = {
            "edge_top": 840,
            "edge_bottom": 920,
            "edge_left": 940,
            "edge_right": 1020,
        }
        ROI5 = {
            "edge_top": 930,
            "edge_bottom": 1000,
            "edge_left": 830,
            "edge_right": 910,
        }
        ROI6 = {
            "edge_top": 930,
            "edge_bottom": 1000,
            "edge_left": 360,
            "edge_right": 440,
        }
        ROI7 = {
            "edge_top": 840,
            "edge_bottom": 920,
            "edge_left": 280,
            "edge_right": 360,
        }
        ROI8 = {
            "edge_top": 360,
            "edge_bottom": 440,
            "edge_left": 280,
            "edge_right": 360,
        }

    # images for object detection
    imcirclist = []
    imcirc1 = Image.fromarray(
        255
        * ArrayDicom[
            ROI1["edge_top"] : ROI1["edge_bottom"],
            ROI1["edge_left"] : ROI1["edge_right"],
        ]
    )
    imcirc1 = imcirc1.resize((imcirc1.width * 10, imcirc1.height * 10), Image.LANCZOS)
    
    list_extent.append(
        (
            (ROI1["edge_left"] * dx / 10),
            (ROI1["edge_right"] * dx / 10),
            (ROI1["edge_bottom"] * dy / 10),
            (ROI1["edge_top"] * dy / 10),
        )
    )

    imcirc2 = Image.fromarray(
        255
        * ArrayDicom[
            ROI2["edge_top"] : ROI2["edge_bottom"],
            ROI2["edge_left"] : ROI2["edge_right"],
        ]
    )
    imcirc2 = imcirc2.resize((imcirc2.width * 10, imcirc2.height * 10), Image.LANCZOS)

    list_extent.append(
        (
            (ROI2["edge_left"] * dx / 10),
            (ROI2["edge_right"] * dx / 10),
            (ROI2["edge_bottom"] * dy / 10),
            (ROI2["edge_top"] * dy / 10),
        )
    )

    imcirc3 = Image.fromarray(
        255
        * ArrayDicom[
            ROI3["edge_top"] : ROI3["edge_bottom"],
            ROI3["edge_left"] : ROI3["edge_right"],
        ]
    )
    imcirc3 = imcirc3.resize((imcirc3.width * 10, imcirc3.height * 10), Image.LANCZOS)
    list_extent.append(
        (
            (ROI3["edge_left"] * dx / 10),
            (ROI3["edge_right"] * dx / 10),
            (ROI3["edge_bottom"] * dy / 10),
            (ROI3["edge_top"] * dy / 10),
        )
    )

    imcirc4 = Image.fromarray(
        255
        * ArrayDicom[
            ROI4["edge_top"] : ROI4["edge_bottom"],
            ROI4["edge_left"] : ROI4["edge_right"],
        ]
    )
    imcirc4 = imcirc4.resize((imcirc4.width * 10, imcirc4.height * 10), Image.LANCZOS)

    list_extent.append(
        (
            (ROI4["edge_left"] * dx / 10),
            (ROI4["edge_right"] * dx / 10),
            (ROI4["edge_bottom"] * dy / 10),
            (ROI4["edge_top"] * dy / 10),
        )
    )

    imcirc5 = Image.fromarray(
        255
        * ArrayDicom[
            ROI5["edge_top"] : ROI5["edge_bottom"],
            ROI5["edge_left"] : ROI5["edge_right"],
        ]
    )
    imcirc5 = imcirc5.resize((imcirc5.width * 10, imcirc5.height * 10), Image.LANCZOS)

    list_extent.append(
        (
            (ROI5["edge_left"] * dx / 10),
            (ROI5["edge_right"] * dx / 10),
            (ROI5["edge_bottom"] * dy / 10),
            (ROI5["edge_top"] * dy / 10),
        )
    )

    imcirc6 = Image.fromarray(
        255
        * ArrayDicom[
            ROI6["edge_top"] : ROI6["edge_bottom"],
            ROI6["edge_left"] : ROI6["edge_right"],
        ]
    )
    imcirc6 = imcirc6.resize((imcirc6.width * 10, imcirc6.height * 10), Image.LANCZOS)

    list_extent.append(
        (
            (ROI6["edge_left"] * dx / 10),
            (ROI6["edge_right"] * dx / 10),
            (ROI6["edge_bottom"] * dy / 10),
            (ROI6["edge_top"] * dy / 10),
        )
    )

    imcirc7 = Image.fromarray(
        255
        * ArrayDicom[
            ROI7["edge_top"] : ROI7["edge_bottom"],
            ROI7["edge_left"] : ROI7["edge_right"],
        ]
    )
    imcirc7 = imcirc7.resize((imcirc7.width * 10, imcirc7.height * 10), Image.LANCZOS)

    list_extent.append(
        (
            (ROI7["edge_left"] * dx / 10),
            (ROI7["edge_right"] * dx / 10),
            (ROI7["edge_bottom"] * dy / 10),
            (ROI7["edge_top"] * dy / 10),
        )
    )

    imcirc8 = Image.fromarray(
        255
        * ArrayDicom[
            ROI8["edge_top"] : ROI8["edge_bottom"],
            ROI8["edge_left"] : ROI8["edge_right"],
        ]
    )
    imcirc8 = imcirc8.resize((imcirc8.width * 10, imcirc8.height * 10), Image.LANCZOS)

    list_extent.append(
        (
            (ROI8["edge_left"] * dx / 10),
            (ROI8["edge_right"] * dx / 10),
            (ROI8["edge_bottom"] * dy / 10),
            (ROI8["edge_top"] * dy / 10),
        )
    )

    imcirclist.append(imcirc1)
    imcirclist.append(imcirc2)
    imcirclist.append(imcirc3)
    imcirclist.append(imcirc4)
    imcirclist.append(imcirc5)
    imcirclist.append(imcirc6)
    imcirclist.append(imcirc7)
    imcirclist.append(imcirc8)

    minSigma=15
    maxSigma=40
    numSigma=10
    thres=0.05
    xdet, ydet = point_detect(imcirclist, minSigma, maxSigma, numSigma, thres)

    profiles = []
    profile1 = np.array(imcirc1, dtype=np.uint8)[:, xdet[0]] / 255
    profile2 = np.array(imcirc2, dtype=np.uint8)[:, xdet[1]] / 255
    profile3 = np.array(imcirc3, dtype=np.uint8)[ydet[2], :] / 255
    profile4 = np.array(imcirc4, dtype=np.uint8)[ydet[3], :] / 255
    profile5 = np.array(imcirc5, dtype=np.uint8)[:, xdet[4]] / 255
    profile6 = np.array(imcirc6, dtype=np.uint8)[:, xdet[5]] / 255
    profile7 = np.array(imcirc7, dtype=np.uint8)[ydet[6], :] / 255
    profile8 = np.array(imcirc8, dtype=np.uint8)[ydet[7], :] / 255

    profiles.append(profile1)
    profiles.append(profile2)
    profiles.append(profile3)
    profiles.append(profile4)
    profiles.append(profile5)
    profiles.append(profile6)
    profiles.append(profile7)
    profiles.append(profile8)



    return profiles, imcirclist, xdet, ydet, list_extent















def roi_sel_GP1(ArrayDicom,ioption,dx,dy):
     # creating the figure extent list for the bib images
    list_extent = []
    if ioption.startswith(('y', 'yeah', 'yes')):
        ROI1 = {"edge_top": 140, "edge_bottom": 310, "edge_left": 435, "edge_right": 605}
        ROI2 = {"edge_top": 300, "edge_bottom": 470, "edge_left": 598, "edge_right": 768}
        ROI3 = {
            "edge_top": 485,
            "edge_bottom": 655,
            "edge_left": 435,
            "edge_right": 605,
        }
        ROI4 = {
            "edge_top": 300,
            "edge_bottom": 470,
            "edge_left": 236,
            "edge_right": 406,
        }
    else:
        ROI1 = {"edge_top": 288, "edge_bottom": 488, "edge_left": 489, "edge_right": 709}
        ROI2 = {"edge_top": 500, "edge_bottom": 700, "edge_left": 684, "edge_right": 889}
        ROI3 = {
            "edge_top": 713,
            "edge_bottom": 913,
            "edge_left": 489,
            "edge_right": 709,
        }
        ROI4 = {
            "edge_top": 500,
            "edge_bottom": 700,
            "edge_left": 280,
            "edge_right": 485,
        }

    # images for object detection
    imcirclist = []
    imcirc1 = Image.fromarray(
        255
        * ArrayDicom[
            ROI1["edge_top"] : ROI1["edge_bottom"],
            ROI1["edge_left"] : ROI1["edge_right"],
        ]
    )
    imcirc1 = imcirc1.resize((imcirc1.width * 10, imcirc1.height * 10), Image.LANCZOS)
    
    list_extent.append(
        (
            (ROI1["edge_left"] * dx / 10),
            (ROI1["edge_right"] * dx / 10),
            (ROI1["edge_bottom"] * dy / 10),
            (ROI1["edge_top"] * dy / 10),
        )
    )

    imcirc2 = Image.fromarray(
        255
        * ArrayDicom[
            ROI2["edge_top"] : ROI2["edge_bottom"],
            ROI2["edge_left"] : ROI2["edge_right"],
        ]
    )
    imcirc2 = imcirc2.resize((imcirc2.width * 10, imcirc2.height * 10), Image.LANCZOS)

    list_extent.append(
        (
            (ROI2["edge_left"] * dx / 10),
            (ROI2["edge_right"] * dx / 10),
            (ROI2["edge_bottom"] * dy / 10),
            (ROI2["edge_top"] * dy / 10),
        )
    )

    imcirc3 = Image.fromarray(
        255
        * ArrayDicom[
            ROI3["edge_top"] : ROI3["edge_bottom"],
            ROI3["edge_left"] : ROI3["edge_right"],
        ]
    )
    imcirc3 = imcirc3.resize((imcirc3.width * 10, imcirc3.height * 10), Image.LANCZOS)
    list_extent.append(
        (
            (ROI3["edge_left"] * dx / 10),
            (ROI3["edge_right"] * dx / 10),
            (ROI3["edge_bottom"] * dy / 10),
            (ROI3["edge_top"] * dy / 10),
        )
    )

    imcirc4 = Image.fromarray(
        255
        * ArrayDicom[
            ROI4["edge_top"] : ROI4["edge_bottom"],
            ROI4["edge_left"] : ROI4["edge_right"],
        ]
    )
    imcirc4 = imcirc4.resize((imcirc4.width * 10, imcirc4.height * 10), Image.LANCZOS)

    list_extent.append(
        (
            (ROI4["edge_left"] * dx / 10),
            (ROI4["edge_right"] * dx / 10),
            (ROI4["edge_bottom"] * dy / 10),
            (ROI4["edge_top"] * dy / 10),
        )
    )


    imcirclist.append(imcirc1)
    imcirclist.append(imcirc2)
    imcirclist.append(imcirc3)
    imcirclist.append(imcirc4)

    minSigma=5
    maxSigma=60
    numSigma=10
    thres=0.05
    xdet, ydet = point_detect(imcirclist, minSigma, maxSigma, numSigma, thres)

    profiles = []
    profile1 = np.array(imcirc1, dtype=np.uint8)[:, xdet[0]] / 255
    profile2 = np.array(imcirc2, dtype=np.uint8)[ydet[1], :] / 255
    profile3 = np.array(imcirc3, dtype=np.uint8)[:, xdet[2]] / 255
    profile4 = np.array(imcirc4, dtype=np.uint8)[ydet[3], :] / 255

    profiles.append(profile1)
    profiles.append(profile2)
    profiles.append(profile3)
    profiles.append(profile4)



    return profiles, imcirclist, xdet, ydet, list_extent
