#############################START LICENSE##########################################
# Copyright (C) 2019 Pedro Martinez
#
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU Affero General Public License as published
# # by the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version (the "AGPL-3.0+").
#
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# # GNU Affero General Public License and the additional terms for more
# # details.
#
# # You should have received a copy of the GNU Affero General Public License
# # along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# # ADDITIONAL TERMS are also included as allowed by Section 7 of the GNU
# # Affero General Public License. These additional terms are Sections 1, 5,
# # 6, 7, 8, and 9 from the Apache License, Version 2.0 (the "Apache-2.0")
# # where all references to the definition "License" are instead defined to
# # mean the AGPL-3.0+.
#
# # You should have received a copy of the Apache-2.0 along with this
# # program. If not, see <http://www.apache.org/licenses/LICENSE-2.0>.
#############################END LICENSE##########################################


###########################################################################################
#
#   Script name: qc-lightrad
#
#   Description: This script performs automated EPID QC of the QC-3 phantom developed in Manitoba.
#   There are other tools out there that do this but generally the ROI are fixed whereas this script
#   aims to dynamically identify them using machine vision and the bibs in the phantom.
#
#   Example usage: python qc-lightrad "/file/"
#
#   Using MED-TEC MT-IAD-1 phantom
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
import utils as u
# matplotlib.use('Qt5Agg')





def point_detect(imcirclist):
    k = 0
    detCenterXRegion = []
    detCenterYRegion = []

    print('Finding bibs in phantom...')
    for img in tqdm(imcirclist):
        grey_img = np.array(img, dtype=np.uint8) #converting the image to grayscale
        blobs_log = blob_log(grey_img, min_sigma=15, max_sigma=50, num_sigma=10, threshold=0.05)
        # print(blobs_log)
        # exit(0)

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

    # plt.figure()
    # plt.imshow(ArrayDicom, extent=extent, origin='upper')
    # plt.imshow(ArrayDicom)
    # plt.xlabel('x distance [cm]')
    # plt.ylabel('y distance [cm]')
    # plt.show()

    if ioption.startswith(('y', 'yeah', 'yes')):
        height, width = ArrayDicom.shape
        ArrayDicom_mod=ArrayDicom[:,width//2-height//2:width//2+height//2]
    else:
        ArrayDicom_mod=ArrayDicom


    #we take a diagonal profile to avoid phantom artifacts
    im_profile = ArrayDicom_mod.diagonal()




    #test to make sure image is displayed correctly bibs are high amplitude against dark background
    ctr_pixel=ArrayDicom_mod[height//2,width//2]
    corner_pixel=ArrayDicom_mod[0,0]

    if ctr_pixel > corner_pixel:
        ArrayDicom = u.range_invert(ArrayDicom)

    ArrayDicom = u.norm01(ArrayDicom)



    #working on transforming the full image and invert it first and go from there.


    if ioption.startswith(('y', 'yeah', 'yes')):
        #images for object detection
        imcirclist = []
        imcirc1 = Image.fromarray(255*ArrayDicom[70:130,280:350 ])
        imcirc1 = imcirc1.resize((imcirc1.width * 10, imcirc1.height * 10), Image.LANCZOS)

        imcirc2 = Image.fromarray(255*ArrayDicom[70:130,680:760 ])
        imcirc2 = imcirc2.resize((imcirc2.width * 10, imcirc2.height * 10), Image.LANCZOS)

        imcirc3 = Image.fromarray(255*ArrayDicom[150:210, 760:830])
        imcirc3 = imcirc3.resize((imcirc3.width * 10, imcirc3.height * 10), Image.LANCZOS)

        imcirc4 = Image.fromarray(255*ArrayDicom[560:620, 760:830])
        imcirc4 = imcirc4.resize((imcirc4.width * 10, imcirc4.height * 10), Image.LANCZOS)

        imcirc5 = Image.fromarray(255*ArrayDicom[640:700,680:760 ])
        imcirc5 = imcirc5.resize((imcirc5.width * 10, imcirc5.height * 10), Image.LANCZOS)

        imcirc6 = Image.fromarray(255*ArrayDicom[640:700, 270:350])
        imcirc6 = imcirc6.resize((imcirc6.width * 10, imcirc6.height * 10), Image.LANCZOS)

        imcirc7 = Image.fromarray(255*ArrayDicom[560:620, 200:270])
        imcirc7 = imcirc7.resize((imcirc7.width * 10, imcirc7.height * 10), Image.LANCZOS)

        imcirc8 = Image.fromarray(255*ArrayDicom[150:220, 200:270])
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
        imcirc1 = Image.fromarray(255*ArrayDicom[280:360, 360:440])
        imcirc1 = imcirc1.resize((imcirc1.width*10,imcirc1.height*10),Image.LANCZOS)
        # imcirc1 = volume[280:360, 360:440]
        # imcirc2 = volume[280:360, 830:910]
        # imcirc2 = misc.imresize(volume[280:360, 830:910], 1000, interp='lanczos', mode='F')
        imcirc2 = Image.fromarray(255*ArrayDicom[280:360, 830:910])
        imcirc2 = imcirc2.resize((imcirc2.width * 10, imcirc2.height * 10), Image.LANCZOS)

        imcirc3 = Image.fromarray(255*ArrayDicom[360:440, 940:1020])
        imcirc3 = imcirc3.resize((imcirc3.width * 10, imcirc3.height * 10), Image.LANCZOS)

        imcirc4 = Image.fromarray(255*ArrayDicom[840:920, 940:1020])
        imcirc4 = imcirc4.resize((imcirc4.width * 10, imcirc4.height * 10), Image.LANCZOS)

        imcirc5 = Image.fromarray(255*ArrayDicom[930:1000, 830:910])
        imcirc5 = imcirc5.resize((imcirc5.width * 10, imcirc5.height * 10), Image.LANCZOS)

        imcirc6 = Image.fromarray(255*ArrayDicom[930:1000, 360:440])
        imcirc6 = imcirc6.resize((imcirc6.width * 10, imcirc6.height * 10), Image.LANCZOS)

        imcirc7 = Image.fromarray(255*ArrayDicom[840:920, 280:360])
        imcirc7 = imcirc7.resize((imcirc7.width * 10, imcirc7.height * 10), Image.LANCZOS)

        imcirc8 = Image.fromarray(255*ArrayDicom[360:440, 280:360])
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
    print('volume=',np.shape(ArrayDicom)[0]/2)

    #creating the page to write the results
    dirname = os.path.dirname(filename)
    print(dirname)

    #tolerance levels to change at will
    tol=1.0 #tolearance level
    act=2.0 #action level

    with PdfPages(dirname + '/' + 'Light-rad_report.pdf') as pdf:
        Page = plt.figure(figsize=(4, 5))
        Page.text(0.45, 0.9, 'Report',size=18)
        kk = 0 #counter for data points
        for profile in profiles:
            value_near,index = u.find_nearest(profile, 0.5) # find the 50% amplitude point


            if k==0 or k==1 or k==4 or k==5: #there are the bibs in the horizontal
                offset_value_y=round(abs((ydet[k]-index)*(dy/10))-3, 2)

                txt = str(offset_value_y)
                print('offset_value_y=',offset_value_y)
                if abs(offset_value_y) <= tol:
                    Page.text(0.1, 0.8 - kk / 10, 'Point' + str(kk+1) + ' offset=' + txt + ' mm',color='g')
                elif abs(offset_value_y) > tol and abs(offset_value_y) <= act:
                    Page.text(0.1, 0.8 - kk / 10, 'Point' + str(kk + 1) + ' offset=' + txt + ' mm', color='y')
                else:
                    Page.text(0.1, 0.8 - kk / 10, 'Point' + str(kk + 1) + ' offset=' + txt + ' mm', color='r')
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
                offset_value_x=round(abs((xdet[k] - index) * (dx/10))-3, 2)

                txt = str(offset_value_x)
                if abs(offset_value_x) <= tol:
                    print('1')
                    Page.text(0.1, 0.8 - kk / 10, 'Point' + str(kk+1) + ' offset=' + txt + ' mm',color='g')
                elif abs(offset_value_x) > tol and abs(offset_value_x) <= act:
                    print('2')
                    Page.text(0.1, 0.8 - kk / 10, 'Point' + str(kk + 1) + ' offset=' + txt + ' mm', color='y')
                else:
                    print('3')
                    Page.text(0.1, 0.8 - kk / 10, 'Point' + str(kk + 1) + ' offset=' + txt + ' mm', color='r')
                kk = kk + 1

                x = np.linspace(0, 0 + (len(profile) * dx* 10), len(profile), endpoint=False)
                ax = fig.add_subplot(4, 2, k + 1)  # plotting all the figures in a single plot
                ax.imshow(np.array(imcirclist[k], dtype=np.uint8)/255)
                ax.scatter(xdet[k], ydet[k],s=30, marker="P", color="y")
                ax.set_title('Bib=' + str(k + 1))
                ax.axvline(index,color="r", linestyle='--')

            k=k+1



        pdf.savefig()
        pdf.savefig(fig)


        # plt.figure()
        # plt.imshow(255*ArrayDicom)
        # plt.show()
        # exit(0)



        # we now need to select a horizontal and a vertical profile to find the edge of the field
        im = Image.fromarray(255 * ArrayDicom)
        # im = im.resize((im.width * 10, im.height * 10), Image.LANCZOS) # we rescale the profile to make it smoother

        if ioption.startswith(('y', 'yeah', 'yes')):

            profilehorz = np.array(im, dtype=np.uint8)[290, :] / 255
            profilevert = np.array(im, dtype=np.uint8)[:, 430] / 255

            # plt.figure()
            # plt.plot(profilehorz)
            # plt.show()
            # exit(0)
            # im_centre = im_centre.resize((im_centre.width * 10, im_centre.height * 10), Image.LANCZOS)

            top_edge,index_top= u.find_nearest(profilevert[0:height//2], 0.5) # finding the edge of the field on the top
            bot_edge,index_bot= u.find_nearest(profilevert[height//2:height], 0.5) # finding the edge of the field on the bottom

            l_edge,index_l = u.find_nearest(profilehorz[0:width//2], 0.5) #finding the edge of the field on the bottom
            r_edge,index_r = u.find_nearest(profilehorz[width//2:width], 0.5) #finding the edge of the field on the right

            print('top_edge','index_top','bot_edge','index_bot')
            print(top_edge,index_top,bot_edge,index_bot)
            print('l_edge', 'index_l', 'r_edge', 'index_r')
            print(l_edge, index_l, r_edge, index_r)

            fig2 = plt.figure(figsize=(7,5))  # this figure will show the vertical and horizontal calculated field size
            ax = fig2.subplots()
            # ax.imshow(volume, extent=extent, origin='upper')
            ax.imshow(ArrayDicom)

            #adding a vertical arrow
            ax.annotate(s='', xy=(430,index_top ), xytext=(430,height//2+index_bot), arrowprops=dict(arrowstyle='<->',color='r')) # example on how to plota double headed arrow
            ax.text(430 + 10, height // 2,'Vfs='+str(round((height//2+index_bot-index_top)*dy/10,2))+'cm',rotation=90, fontsize=14, color='r')

            #adding a horizontal arrow
            ax.annotate(s='', xy=(index_l,290), xytext=(width // 2 + index_r,290),
                        arrowprops=dict(arrowstyle='<->',color='r'))  # example on how to plota double headed arrow
            ax.text(width//2, 290-5, 'Hfs='+str(round((width // 2 + index_r-index_l)*dx/10,2))+'cm', rotation=0, fontsize=14, color='r')


            # plt.xlabel('x distance [cm]')
            # plt.ylabel('y distance [cm]')


        else:
            profilehorz = np.array(im, dtype=np.uint8)[470, :] / 255
            profilevert = np.array(im, dtype=np.uint8)[:, 540] / 255

            top_edge, index_top = u.find_nearest(profilevert[0:height // 2],
                                               0.5)  # finding the edge of the field on the top
            bot_edge, index_bot = u.find_nearest(profilevert[height // 2:height],
                                               0.5)  # finding the edge of the field on the bottom

            l_edge, index_l = u.find_nearest(profilehorz[0:width // 2],
                                           0.5)  # finding the edge of the field on the bottom
            r_edge, index_r = u.find_nearest(profilehorz[width // 2:width],
                                           0.5)  # finding the edge of the field on the right

            # print('top_edge', 'index_top', 'bot_edge', 'index_bot')
            # print(top_edge, index_top, bot_edge, index_bot)
            # print('l_edge', 'index_l', 'r_edge', 'index_r')
            # print(l_edge, index_l, r_edge, index_r)

            fig2 = plt.figure(figsize=(7, 5))  # this figure will show the vertical and horizontal calculated field size
            ax = fig2.subplots()
            # ax.imshow(volume, extent=extent, origin='upper')
            ax.imshow(ArrayDicom)

            # adding a vertical arrow
            ax.annotate(s='', xy=(540, index_top), xytext=(540, height // 2 + index_bot),
                        arrowprops=dict(arrowstyle='<->'))  # example on how to plota double headed arrow
            ax.text(540+10, height // 2 ,
                   'Vfs='+ str(round((height // 2 + index_bot - index_top) * dy / 10, 2)) + 'cm', rotation=90, fontsize=14)

            # adding a horizontal arrow
            ax.annotate(s='', xy=(index_l, 470), xytext=(width // 2 + index_r, 470),
                        arrowprops=dict(arrowstyle='<->'))  # example on how to plota double headed arrow
            ax.text(width // 2, 470-5, 'Hfs='+str(round((width // 2 + index_r - index_l) * dx / 10, 2)) + 'cm', rotation=0,
                    fontsize=14)

            # plt.xlabel('x distance [cm]')
            # plt.ylabel('y distance [cm]')



        # plt.show()





        pdf.savefig(fig2)










    # # Normal mode:
    # print()
    # print("Directory folder.........:", dirname)
    # print("Storage type.....:", dataset.SOPClassUID)
    # print()
    #
    # pat_name = dataset.PatientName
    # display_name = pat_name.family_name + ", " + pat_name.given_name
    # print("Patient's name...:", display_name)
    # print("Patient id.......:", dataset.PatientID)
    # print("Modality.........:", dataset.Modality)
    # print("Study Date.......:", dataset.StudyDate)
    # print("Gantry angle......", dataset.GantryAngle)
















































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









