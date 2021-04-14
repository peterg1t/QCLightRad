import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
from skimage.feature import blob_log
import matplotlib.pyplot as plt



def point_detect(imcirclist, minSigma, maxSigma, numSigma, thres):
    k = 0
    detCenterXRegion = []
    detCenterYRegion = []

    print("Finding bibs in phantom...")
    for img in tqdm(imcirclist):
        
        # #Image pre-processing to detect low amplitude bibs
        # img=img.convert(mode='L')
        # #image brightness enhancer
        # enhancer = ImageEnhance.Brightness(img)
        # img = enhancer.enhance(3.5)
        # img.save('img_enh_b','PNG')
        # #image brightness enhancer
        # enhancer = ImageEnhance.Contrast(img)
        # img = enhancer.enhance(1.5)

        grey_img = np.array(img, dtype=np.uint8)  # converting the image to a numpy matrix
        blobs_log = blob_log(
            grey_img, min_sigma=minSigma, max_sigma=maxSigma, num_sigma=numSigma, threshold=thres
        )

        centerXRegion = []
        centerYRegion = []
        centerRRegion = []
        grey_ampRegion = []
        for blob in blobs_log:
            y, x, r = blob
            center = (int(x), int(y))
            centerXRegion.append(x)
            centerYRegion.append(y)
            centerRRegion.append(r)
            grey_ampRegion.append(grey_img[int(y), int(x)])
            radius = int(r)
            print('center=', center, 'radius=', radius, 'pixel_val',grey_img[int(y), int(x)])

        
        xindx = int(centerXRegion[np.argmin(grey_ampRegion)])
        yindx = int(centerYRegion[np.argmin(grey_ampRegion)])
        # rindx = int(centerRRegion[np.argmin(grey_ampRegion)])

        # fig,ax = plt.subplots()
        # ax.imshow(img)
        # ax.scatter(xindx,yindx)
        # plt.show(block=True)


        detCenterXRegion.append(xindx)
        detCenterYRegion.append(yindx)

        k = k + 1

    return detCenterXRegion, detCenterYRegion








def roi_sel_FC2(ArrayDicom,ioption,dx,dy):
     # creating the figure extent list for the bib images
    list_extent = []
    if ioption==1:  #these are the ROI for a clinac
        print('Clinac machine detected...')
        ROI1 = {"edge_top": 66, "edge_bottom": 166, "edge_left": 195, "edge_right": 295}
        ROI2 = {"edge_top": 66, "edge_bottom": 166, "edge_left": 728, "edge_right": 828}
        ROI3 = {"edge_top": 582,"edge_bottom": 682,"edge_left": 728,"edge_right": 828}
        ROI4 = {"edge_top": 582,"edge_bottom": 682,"edge_left": 195,"edge_right": 295}
    elif ioption==2:
        print('TrueBeam Edge machine detected...')
        ROI1 = {   #these are the ROI for a TrueBeam Edge
                "edge_top": 66,         #y1
                "edge_bottom": 166,     #y2
                "edge_left": 195,       #x1
                "edge_right": 295,      #x2
            }
        ROI2 = {
               "edge_top": 66,
               "edge_bottom": 166,
               "edge_left": 728,
               "edge_right": 828,
           }
        ROI3 = {
               "edge_top": 582,
               "edge_bottom": 682,
               "edge_left": 728,
               "edge_right": 828,
           }
        ROI4 = {
               "edge_top": 582,
               "edge_bottom": 682,
               "edge_left": 195,
               "edge_right": 295,
           }
    elif ioption == 3:
        print('Varian XI machine detected...')
        ROI1 = {  # these are the ROI for a TrueBeam XI
            "edge_top": 289,  # y1
            "edge_bottom": 394,  # y2
            "edge_left": 299,  # x1
            "edge_right": 404,  # x2
        }
        ROI2 = {
            "edge_top": 289,
            "edge_bottom": 394,
            "edge_left": 880,
            "edge_right": 985,
        }
        ROI3 = {
            "edge_top": 880,
            "edge_bottom": 985,
            "edge_left": 880,
            "edge_right": 985,
        }
        ROI4 = {
            "edge_top": 880,
            "edge_bottom": 985,
            "edge_left": 299,
            "edge_right": 404,
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



    minSigma=15
    maxSigma=40
    numSigma=10
    thres=0.05


    
    xdet, ydet = point_detect(imcirclist, minSigma, maxSigma, numSigma, thres)

    profiles = []
    profile1a = np.array(imcirc1, dtype=np.uint8)[:, xdet[0]] / 255
    profile1b = np.array(imcirc1, dtype=np.uint8)[ydet[0],:] / 255
    profile2a = np.array(imcirc2, dtype=np.uint8)[:, xdet[1]] / 255
    profile2b = np.array(imcirc2, dtype=np.uint8)[ydet[1],:] / 255
    profile3a = np.array(imcirc3, dtype=np.uint8)[:, xdet[2]] / 255
    profile3b = np.array(imcirc3, dtype=np.uint8)[ydet[2],:] / 255
    profile4a = np.array(imcirc4, dtype=np.uint8)[:, xdet[3]] / 255
    profile4b = np.array(imcirc4, dtype=np.uint8)[ydet[3],:] / 255

    profiles.append(profile1a)
    profiles.append(profile1b)
    profiles.append(profile2a)
    profiles.append(profile2b)
    profiles.append(profile3a)
    profiles.append(profile3b)
    profiles.append(profile4a)
    profiles.append(profile4b)


    return profiles, imcirclist, xdet, ydet, list_extent















def roi_sel_IsoAlign(ArrayDicom,ioption,dx,dy):
     # creating the figure extent list for the bib images
    list_extent = []
    if ioption == 1:
        print('Clinac machine detected...')
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
    elif ioption == 2:
        print('TrueBeam Edge machine detected...')
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
    elif ioption == 3:
        print('TrueBeam Edge machine detected...')
        ROI1 = {
            "edge_top": 266,
            "edge_bottom": 366,
            "edge_left": 350,
            "edge_right": 450,
        }
        ROI2 = {
            "edge_top": 266,
            "edge_bottom": 366,
            "edge_left": 828,
            "edge_right": 928,
        }
        ROI3 = {
            "edge_top": 351,
            "edge_bottom": 451,
            "edge_left": 918,
            "edge_right": 1018,
        }
        ROI4 = {
            "edge_top": 827,
            "edge_bottom": 927,
            "edge_left": 919,
            "edge_right": 1019,
        }
        ROI5 = {
            "edge_top": 917,
            "edge_bottom": 1017,
            "edge_left": 829,
            "edge_right": 929,
        }
        ROI6 = {
            "edge_top": 912,
            "edge_bottom": 1012,
            "edge_left": 353,
            "edge_right": 453,
        }
        ROI7 = {
            "edge_top": 832,
            "edge_bottom": 932,
            "edge_left": 265,
            "edge_right": 365,
        }
        ROI8 = {
            "edge_top": 354,
            "edge_bottom": 454,
            "edge_left": 261,
            "edge_right": 361,
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


    # tot_image=Image.fromarray(255* ArrayDicom)
    # tot_image.show()
    # for img in imcirclist:
    #     print(img)
    #     img.show()
    # exit(0)

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
    if ioption == 1:
        print('Clinac machine detected...')
        ROI1 = {"edge_top": 165, "edge_bottom": 265, "edge_left": 462, "edge_right": 562}
        ROI2 = {"edge_top": 339, "edge_bottom": 439, "edge_left": 635, "edge_right": 735}
        ROI3 = {
            "edge_top": 506,
            "edge_bottom": 606,
            "edge_left": 462,
            "edge_right": 562,
        }
        ROI4 = {
            "edge_top": 339,
            "edge_bottom": 439,
            "edge_left": 292,
            "edge_right": 392,
        }
    elif ioption == 2:
        print('TrueBeam Edge machine detected...')
        ROI1 = {"edge_top": 344, "edge_bottom": 444, "edge_left": 543, "edge_right": 643}
        ROI2 = {"edge_top": 545, "edge_bottom": 645, "edge_left": 740, "edge_right": 840}
        ROI3 = {
            "edge_top": 743,
            "edge_bottom": 843,
            "edge_left": 543,
            "edge_right": 643,
        }
        ROI4 = {
            "edge_top": 550,
            "edge_bottom": 650,
            "edge_left": 345,
            "edge_right": 445,
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
