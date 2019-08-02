###########################################################################################
#
#   Script name: xqc-LRCircleDet.py
#
#   Description: 
#
#   Example usage: python xqc-LRCircleDet.py "/file/"
#
#   Author: Pedro Martinez
#   pedro.enrique.83@gmail.com
#   5877000722
#   Date:2019-04-09
#
###########################################################################################



import os
import sys
import pydicom
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
from skimage.feature import blob_dog, blob_log, blob_doh
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal, misc
from PIL import Image
from tqdm import tqdm
import argparse
# matplotlib.use('Qt5Agg')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx
    # return array[idx]


def point_detect(imcirclist):
    k = 0
    detCenterXRegion = []
    detCenterYRegion = []

    print('Finding bibs in phantom...')
    for img in tqdm(imcirclist):
        grey_img = np.array(img, dtype=np.uint8) #converting the image to grayscale
        blobs_log = blob_log(grey_img, min_sigma=15, max_sigma=50, num_sigma=10, threshold=0.05)

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
            # print('center=', center, 'radius=', radius, 'value=', img[center], grey_img[center])

        xindx = int(centerXRegion[np.argmin(grey_ampRegion)])
        yindx = int(centerYRegion[np.argmin(grey_ampRegion)])
        rindx = int(centerRRegion[np.argmin(grey_ampRegion)])


        detCenterXRegion.append(xindx)
        detCenterYRegion.append(yindx)


        k = k + 1


    return detCenterXRegion,detCenterYRegion








def read_dicom(filename,ioption):
    dataset = pydicom.dcmread(filename)
    ArrayDicom = np.zeros((dataset.Rows, dataset.Columns), dtype=dataset.pixel_array.dtype)
    ArrayDicom = dataset.pixel_array
    SID = dataset.RTImageSID
    print('array_shape=',np.shape(ArrayDicom))
    height=np.shape(ArrayDicom)[0]
    width = np.shape(ArrayDicom)[1]
    dx = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[0]) / 1000)
    dy = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[1]) / 1000)
    print("pixel spacing row [mm]=", dx)
    print("pixel spacing col [mm]=", dy)

    #creating the figure extent based on the image dimensions, we divide by 10 to get the units in cm
    extent = (0, 0 + (ArrayDicom.shape[0] * dx/10),
              0, 0 + (ArrayDicom.shape[1] * dy/10))

    plt.figure()
    plt.imshow(ArrayDicom, extent=extent, origin='upper')
    plt.xlabel('x distance [cm]')
    plt.ylabel('y distance [cm]')
    plt.show()

    if ioption.startswith(('y', 'yeah', 'yes')):
        height, width = ArrayDicom.shape
        ArrayDicom_mod=ArrayDicom[:,width//2-height//2:width//2+height//2]
    else:
        ArrayDicom_mod=ArrayDicom


    #we take a diagonal profile to avoid phantom artifacts
    im_profile = ArrayDicom_mod.diagonal()

    min_val = np.amin(im_profile)  # normalizing
    volume = np.int16(np.subtract(ArrayDicom , min_val))
    volume = volume / np.amax(volume)




    #working on transforming the full image and invert it first and go from there.


    if ioption.startswith(('y', 'yeah', 'yes')):
        #images for object detection
        imcirclist = []
        imcirc1 = Image.fromarray(255*volume[70:130,280:350 ])
        imcirc1 = imcirc1.resize((imcirc1.width * 10, imcirc1.height * 10), Image.LANCZOS)

        imcirc2 = Image.fromarray(255*volume[70:130,680:760 ])
        imcirc2 = imcirc2.resize((imcirc2.width * 10, imcirc2.height * 10), Image.LANCZOS)

        imcirc3 = Image.fromarray(255*volume[150:210, 760:830])
        imcirc3 = imcirc3.resize((imcirc3.width * 10, imcirc3.height * 10), Image.LANCZOS)

        imcirc4 = Image.fromarray(255*volume[560:620, 760:830])
        imcirc4 = imcirc4.resize((imcirc4.width * 10, imcirc4.height * 10), Image.LANCZOS)

        imcirc5 = Image.fromarray(255*volume[640:700,680:760 ])
        imcirc5 = imcirc5.resize((imcirc5.width * 10, imcirc5.height * 10), Image.LANCZOS)

        imcirc6 = Image.fromarray(255*volume[640:700, 270:350])
        imcirc6 = imcirc6.resize((imcirc6.width * 10, imcirc6.height * 10), Image.LANCZOS)

        imcirc7 = Image.fromarray(255*volume[560:620, 200:270])
        imcirc7 = imcirc7.resize((imcirc7.width * 10, imcirc7.height * 10), Image.LANCZOS)

        imcirc8 = Image.fromarray(255*volume[150:220, 200:270])
        imcirc8 = imcirc8.resize((imcirc8.width * 10, imcirc8.height * 10), Image.LANCZOS)


        imcirclist.append(imcirc1)
        imcirclist.append(imcirc2)
        imcirclist.append(imcirc3)
        imcirclist.append(imcirc4)
        imcirclist.append(imcirc5)
        imcirclist.append(imcirc6)
        imcirclist.append(imcirc7)
        imcirclist.append(imcirc8)

        xdet, ydet = point_detect(imcirclist)

        profiles = []
        profile1 = np.array(imcirc1, dtype=np.uint8)[:,xdet[0]]/255
        profile2 = np.array(imcirc2, dtype=np.uint8)[:,xdet[1]]/255
        profile3 = np.array(imcirc3, dtype=np.uint8)[ydet[2],:]/255
        profile4 = np.array(imcirc4, dtype=np.uint8)[ydet[3],:]/255
        profile5 = np.array(imcirc5, dtype=np.uint8)[:,xdet[4]]/255
        profile6 = np.array(imcirc6, dtype=np.uint8)[:,xdet[5]]/255
        profile7 = np.array(imcirc7, dtype=np.uint8)[ydet[6],:]/255
        profile8 = np.array(imcirc8, dtype=np.uint8)[ydet[7],:]/255

        profiles.append(profile1)
        profiles.append(profile2)
        profiles.append(profile3)
        profiles.append(profile4)
        profiles.append(profile5)
        profiles.append(profile6)
        profiles.append(profile7)
        profiles.append(profile8)

    else:
        imcirclist = []
        # imcirc1 = misc.imresize(volume[280:360, 360:440],1000, interp='lanczos', mode='F')
        imcirc1 = Image.fromarray(255*volume[280:360, 360:440])
        imcirc1 = imcirc1.resize((imcirc1.width*10,imcirc1.height*10),Image.LANCZOS)
        # imcirc1 = volume[280:360, 360:440]
        # imcirc2 = volume[280:360, 830:910]
        # imcirc2 = misc.imresize(volume[280:360, 830:910], 1000, interp='lanczos', mode='F')
        imcirc2 = Image.fromarray(255*volume[280:360, 830:910])
        imcirc2 = imcirc2.resize((imcirc2.width * 10, imcirc2.height * 10), Image.LANCZOS)

        imcirc3 = Image.fromarray(255*volume[360:440, 940:1020])
        imcirc3 = imcirc3.resize((imcirc3.width * 10, imcirc3.height * 10), Image.LANCZOS)

        imcirc4 = Image.fromarray(255*volume[840:920, 940:1020])
        imcirc4 = imcirc4.resize((imcirc4.width * 10, imcirc4.height * 10), Image.LANCZOS)

        imcirc5 = Image.fromarray(255*volume[930:1000, 830:910])
        imcirc5 = imcirc5.resize((imcirc5.width * 10, imcirc5.height * 10), Image.LANCZOS)

        imcirc6 = Image.fromarray(255*volume[930:1000, 360:440])
        imcirc6 = imcirc6.resize((imcirc6.width * 10, imcirc6.height * 10), Image.LANCZOS)

        imcirc7 = Image.fromarray(255*volume[840:920, 280:360])
        imcirc7 = imcirc7.resize((imcirc7.width * 10, imcirc7.height * 10), Image.LANCZOS)

        imcirc8 = Image.fromarray(255*volume[360:440, 280:360])
        imcirc8 = imcirc8.resize((imcirc8.width * 10, imcirc8.height * 10), Image.LANCZOS)


        imcirclist.append(imcirc1)
        imcirclist.append(imcirc2)
        imcirclist.append(imcirc3)
        imcirclist.append(imcirc4)
        imcirclist.append(imcirc5)
        imcirclist.append(imcirc6)
        imcirclist.append(imcirc7)
        imcirclist.append(imcirc8)

        xdet, ydet = point_detect(imcirclist)
        print('location->',xdet,ydet)


        # we need to find a profile along the middle point and find the edge of the field at 80% then we
        # measure the distance from the field to the bib


        #converting back to array
        profiles = []
        profile1 = np.array(imcirc1, dtype=np.uint8)[:,xdet[0]]/255
        profile2 = np.array(imcirc2, dtype=np.uint8)[:,xdet[1]]/255
        profile3 = np.array(imcirc3, dtype=np.uint8)[ydet[2],:]/255
        profile4 = np.array(imcirc4, dtype=np.uint8)[ydet[3],:]/255
        profile5 = np.array(imcirc5, dtype=np.uint8)[:,xdet[4]]/255
        profile6 = np.array(imcirc6, dtype=np.uint8)[:,xdet[5]]/255
        profile7 = np.array(imcirc7, dtype=np.uint8)[ydet[6],:]/255
        profile8 = np.array(imcirc8, dtype=np.uint8)[ydet[7],:]/255

        profiles.append(profile1)
        profiles.append(profile2)
        profiles.append(profile3)
        profiles.append(profile4)
        profiles.append(profile5)
        profiles.append(profile6)
        profiles.append(profile7)
        profiles.append(profile8)


    k=0
    fig = plt.figure(figsize=(7, 9))# this figure will hold the bibs
    plt.subplots_adjust(hspace=0.35)

    #getting a profile to extract max value to normalize
    print('volume=',np.shape(volume)[0]/2)

    #creating the page to write the results
    dirname = os.path.dirname(filename)
    print(dirname)
    with PdfPages(dirname + '/' + 'Light-rad_report.pdf') as pdf:
        Page = plt.figure(figsize=(4, 5))
        Page.text(0.45, 0.9, 'Report',size=18)
        kk = 0 #counter for data points
        for profile in profiles:
            value_near,index = find_nearest(profile, 0.5)

            if k==0 or k==1 or k==4 or k==5: #there are the bibs in thehorizontal
                print(value_near,index)
                print('k=',k,'ydet=',ydet[k],'index=',index,'delta=',(ydet[k]-index),'px','delta=',
                      abs((ydet[k]-index)*dy/10)-3,'mm','dy=',dy/10)

                txt = str(round(abs((ydet[k]-index)*dy/10)-3, 2))
                Page.text(0.1, 0.8 - kk / 10, 'Point' + str(kk+1) + ' delta=' + txt + ' mm')
                kk = kk + 1

                y = np.linspace(0, 0 + (len(profile) * dy *10 ), len(profile), endpoint=False)
                ax = fig.add_subplot(4, 2, k + 1)  # plotting all the figures in a single plot
                ax.imshow(np.array(imcirclist[k], dtype=np.uint8)/255)
                ax.scatter(xdet[k], ydet[k],s=30, marker="P", color="y")
                ax.set_title('Bib=' + str(k + 1))
                ax.axhline(index,color="r", linestyle='--')
                # plt.figure()
                # plt.scatter(y,profile)
                # plt.scatter(index*dy*10,value_near)
                # plt.axvline((ydet[k]-1)*dy*10)
                # plt.show()
            else:
                print(value_near, index)
                print('ydet=', xdet[k], 'index=', index, 'delta=', (xdet[k] - index), 'px', 'delta=',
                      abs((xdet[k] - index) * dx/10)-3, 'mm', 'dx=', dx/10)

                txt = str(round(abs((xdet[k] - index) * dx/10)-3, 2))
                Page.text(0.1, 0.8 - kk / 10, 'Point' + str(kk+1) + ' delta=' + txt + ' mm')
                kk = kk + 1

                x = np.linspace(0, 0 + (len(profile) * dx* 10), len(profile), endpoint=False)
                ax = fig.add_subplot(4, 2, k + 1)  # plotting all the figures in a single plot
                ax.imshow(np.array(imcirclist[k], dtype=np.uint8)/255)
                ax.scatter(xdet[k], ydet[k],s=30, marker="P", color="y")
                ax.set_title('Bib=' + str(k + 1))
                ax.axvline(index,color="r", linestyle='--')

                # plt.figure()
                # plt.scatter(x, profile)
                # plt.scatter(index * dx*10, value_near)
                # plt.axvline((xdet[k]-1) * dx*10)
                # plt.show()


            k=k+1



        pdf.savefig()
        pdf.savefig(fig)




        fig2 = plt.figure(figsize=(7, 9))  # this figure will show the vertical and horizontal calculated field size
        ax = fig2.add_subplot()
        ax.imshow(volume, extent=extent, origin='upper')
        # plt.xlabel('x distance [cm]')
        # plt.ylabel('y distance [cm]')

        # we now need to select a horizontal and a vertical profile to find the edge of the field
        im = Image.fromarray(255 * volume[:,:])
        im = im.resize((im.width * 10, im.height * 10), Image.LANCZOS) # we rescale the profile to make it smoother
        profilehorz = np.array(im, dtype=np.uint8)["value here", :] / 255
        profilevert = np.array(im, dtype=np.uint8)[:, "value here"] / 255

        # top_edge,index_top= find_nearest(profile, 0.5) # finding the edge of the field on the top
        # bot_edge,index_bot= find_nearest(profile, 0.5) # finding the edge of the field on the bottom

        # l_edge,index_r = find_nearest(profile, 0.5) #finding the edge of the field on the bottom
        # r_edge,index_l = find_nearest(profile, 0.5) #finding the edge of the field on the right

        # plt.annotate(s='', xy=(1, 1), xytext=(0, 0), arrowprops=dict(arrowstyle='<->')) # example on how to plota double headed arrow

        pdf.savefig(fig2)










    # Normal mode:
    print()
    print("Directory folder.........:", dirname)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name...:", display_name)
    print("Patient id.......:", dataset.PatientID)
    print("Modality.........:", dataset.Modality)
    print("Study Date.......:", dataset.StudyDate)
    print("Gantry angle......", dataset.GantryAngle)
















































#Data ipunt
# try:
#     filename = str(sys.argv[1])
#     print('filename=',filename)
# except:
#     print('Please enter a valid directory name')
#     print("Use the following command to run this script")
#     print("python qc-LRCircleDet.py \"[filename]\" ")


while True:  # example of infinite loops using try and except to catch only numbers
    line = input('Are these files from a clinac [yes(y)/no(n)]> ')
    try:
        ##        if line == 'done':
        ##            break
        ioption = str(line.lower())
        if ioption.startswith(('y', 'yeah', 'yes', 'n', 'no', 'nope')):
            break

    except:
        print('Please enter a valid option:')


parser = argparse.ArgumentParser()
parser.add_argument('file',type=str,help="Input the Light/Rad file")
args=parser.parse_args()

filename=args.file



read_dicom(filename,ioption)






























# **************EXAMPLES***************************
# ## Select boxes
# bboxes = []
# colors = []
#
# # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# # So we will call this function in a loop till we are done selecting all objects
# while True:
#     # draw bounding boxes over objects
#     # selectROI's default behaviour is to draw box starting from the center
#     # when fromCenter is set to false, you can draw box starting from top left corner
#     bbox = cv2.selectROI('MultiTracker', LightRadEq)
#     bboxes.append(bbox)
#     colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
#     print("Press q to quit selecting boxes and start tracking")
#     print("Press any other key to select next object")
#     k = cv2.waitKey(0) & 0xFF
#     if (k == 113):  # q is pressed
#         break
#
# print('Selected bounding boxes {}'.format(bboxes))

# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# for (x, y, w, h) in faces:
#     cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     roi_gray = gray[y:y + h, x:x + w]
#     roi_color = img[y:y + h, x:x + w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex, ey, ew, eh) in eyes:
#         cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# **************EXAMPLES***************************









