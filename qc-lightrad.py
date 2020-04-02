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

import argparse
import os
from datetime import datetime
from sys import platform
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from skimage.feature import blob_log
import pydicom
import roi_sel as roi
import utils as u
import inquirer
from timeit import default_timer as timer


def point_detect(imcirclist):
    k = 0
    detCenterXRegion = []
    detCenterYRegion = []

    print("Finding bibs in phantom...")
    for img in tqdm(imcirclist):
        grey_img = np.array(img, dtype=np.uint8)  # converting the image to grayscale
        blobs_log = blob_log(
            grey_img, min_sigma=15, max_sigma=40, num_sigma=10, threshold=0.05
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


def read_dicom(filenm):
    dataset = pydicom.dcmread(filenm)
    now = datetime.now()

    ArrayDicom = np.zeros(
        (dataset.Rows, dataset.Columns), dtype=dataset.pixel_array.dtype
    )
    ArrayDicom = dataset.pixel_array
    SID = dataset.RTImageSID
    print("array_shape=", np.shape(ArrayDicom))
    height = np.shape(ArrayDicom)[0]
    width = np.shape(ArrayDicom)[1]
    dx = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[0]) / 1000)
    dy = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[1]) / 1000)
    print("pixel spacing row [mm]=", dx)
    print("pixel spacing col [mm]=", dy)

    # creating the figure extent based on the image dimensions, we divide by 10 to get the units in cm
    extent = (
        0,
        0 + (ArrayDicom.shape[1] * dx / 10),
        0 + (ArrayDicom.shape[0] * dy / 10),
        0,
    )

    # creating the figure extent list for the bib images
    list_extent = []

    # plt.figure()
    # plt.imshow(ArrayDicom, extent=extent, origin='upper')
    # plt.imshow(ArrayDicom)
    # plt.xlabel('x distance [cm]')
    # plt.ylabel('y distance [cm]')
    # plt.show()

    print('np.shape_dim0',np.shape(ArrayDicom)[0])
    if np.shape(ArrayDicom)[0] == 768: #Clinac
        ioptn=1
    elif np.shape(ArrayDicom)[0] == 1190: #Edge EPID
        ioptn=2
    elif np.shape(ArrayDicom)[0] == 1280: ##Varian XI (Edmonton)
        ioptn=3



    if ioptn == 1:
        height, width = ArrayDicom.shape
        ArrayDicom_mod = ArrayDicom[:, width // 2 - height // 2 : width // 2 + height // 2]
    else:
        ArrayDicom_mod = ArrayDicom

    # we take a diagonal profile to avoid phantom artifacts
    # im_profile = ArrayDicom_mod.diagonal()

    # test to make sure image is displayed correctly bibs are high amplitude against dark background
    ctr_pixel = ArrayDicom_mod[height // 2, width // 2]
    corner_pixel = ArrayDicom_mod[0, 0]

    if ctr_pixel > corner_pixel:
        ArrayDicom = u.range_invert(ArrayDicom)

    ArrayDicom = u.norm01(ArrayDicom)






    #Here we need to select the correct ROIs for the image processing since in the FC-2 phantom the bibs are in the corners we will select more than 1 ROI per-bib maybe??



    questions = [
        inquirer.List(
            "type",
            message="Select the phantom",
            choices=[
                "IsoAlign",
                "FC-2",
                "GP-1",
            ],
        )
    ]
    answers = inquirer.prompt(questions)
    print(answers["type"])

    if answers["type"] == "IsoAlign":
        profiles, imcirclist, xdet, ydet, list_extent = roi.roi_sel_IsoAlign(ArrayDicom,ioptn,dx,dy)
    elif answers["type"] == "FC-2":
        profiles, imcirclist, xdet, ydet, list_extent = roi.roi_sel_FC2(ArrayDicom,ioptn,dx,dy)
    elif answers["type"] == "GP-1":
        start = timer()
        profiles, imcirclist, xdet, ydet, list_extent = roi.roi_sel_GP1(ArrayDicom,ioptn,dx,dy)
        dt = timer() - start
        print( "File processed in %f s" % dt)



    
    # tolerance levels to change at will
    tol = 1.0  # tolearance level
    act = 2.0  # action level



    k = 0
    fig = plt.figure(figsize=(8, 12))  # this figure will hold the bibs
    plt.subplots_adjust(hspace=0.35)

    # creating the page to write the results
    dirname = os.path.dirname(filenm)


    if platform == "linux":
        output_flnm=dirname+ "/"+ now.strftime("%d-%m-%Y_%H:%M_")+ dataset[0x0008, 0x1010].value+ "_Lightrad_report.pdf"
    elif platform == "win32":
        output_flnm=dataset[0x0008, 0x1010].value+ "_Lightrad_report.pdf"

    print('Writing to:',output_flnm)

    with PdfPages(
        output_flnm
    ) as pdf:
        Page = plt.figure(figsize=(4, 5))
        Page.text(0.45, 0.9, "Report", size=18)
        if answers["type"] == "IsoAlign":
            phantom_distance = 3.0  # distance from the bib to the edge of the phantom in mm
            kk = 0  # counter for data points
            for profile in profiles:
                _, index = u.find_nearest(profile, 0.5)  # find the 50% amplitude point
                # value_near, index = find_nearest(profile, 0.5) # find the 50% amplitude point
                if (  # pylint: disable = consider-using-in
                    k == 0 or k == 1 or k == 4 or k == 5
                ):  # there are the bibs in the horizontal
                    offset_value_y = round(
                        abs((ydet[k] - index) * (dy / 10)) - phantom_distance, 2
                    )

                    txt = str(offset_value_y)
                    # print('offset_value_y=', offset_value_y)
                    if abs(offset_value_y) <= tol:
                        Page.text(
                            0.1,
                            0.8 - kk / 10,
                            "Point" + str(kk + 1) + " offset=" + txt + " mm",
                            color="g",
                        )
                    elif abs(offset_value_y) > tol and abs(offset_value_y) <= act:
                        Page.text(
                            0.1,
                            0.8 - kk / 10,
                            "Point" + str(kk + 1) + " offset=" + txt + " mm",
                            color="y",
                        )
                    else:
                        Page.text(
                            0.1,
                            0.8 - kk / 10,
                            "Point" + str(kk + 1) + " offset=" + txt + " mm",
                            color="r",
                        )
                    kk = kk + 1

                    ax = fig.add_subplot(
                        4, 2, k + 1
                    )  # plotting all the figures in a single plot

                    ax.imshow(
                        np.array(imcirclist[k], dtype=np.uint8) / 255,
                        extent=list_extent[k],
                        origin="upper",
                    )
                    ax.scatter(
                        list_extent[k][0] + xdet[k] * dx / 100,
                        list_extent[k][3] + ydet[k] * dy / 100,
                        s=30,
                        marker="P",
                        color="y",
                    )
                    ax.set_title("Bib=" + str(k + 1))
                    ax.axhline(
                        list_extent[k][3] + index * dy / 100, color="r", linestyle="--"
                    )
                    ax.set_xlabel("x distance [cm]")
                    ax.set_ylabel("y distance [cm]")
                else:
                    offset_value_x = round(
                        abs((xdet[k] - index) * (dx / 10)) - phantom_distance, 2
                    )

                    txt = str(offset_value_x)
                    if abs(offset_value_x) <= tol:
                        # print('1')
                        Page.text(
                            0.1,
                            0.8 - kk / 10,
                            "Point" + str(kk + 1) + " offset=" + txt + " mm",
                            color="g",
                        )
                    elif abs(offset_value_x) > tol and abs(offset_value_x) <= act:
                        # print('2')
                        Page.text(
                            0.1,
                            0.8 - kk / 10,
                            "Point" + str(kk + 1) + " offset=" + txt + " mm",
                            color="y",
                        )
                    else:
                        # print('3')xdet[0]
                        Page.text(
                            0.1,
                            0.8 - kk / 10,
                            "Point" + str(kk + 1) + " offset=" + txt + " mm",
                            color="r",
                        )
                    kk = kk + 1

                    ax = fig.add_subplot(
                        4, 2, k + 1
                    )  # plotting all the figures in a single plot

                    ax.imshow(
                        np.array(imcirclist[k], dtype=np.uint8) / 255,
                        extent=list_extent[k],
                        origin="upper",
                    )
                    ax.scatter(
                        list_extent[k][0] + xdet[k] * dx / 100,
                        list_extent[k][3] + ydet[k] * dy / 100,
                        s=30,
                        marker="P",
                        color="y",
                    )
                    ax.set_title("Bib=" + str(k + 1))
                    ax.axvline(
                        list_extent[k][0] + index * dx / 100, color="r", linestyle="--"
                    )
                    ax.set_xlabel("x distance [cm]")
                    ax.set_ylabel("y distance [cm]")

                k = k + 1
        elif answers["type"] == "GP-1":
            phantom_distance = 10.0  # distance from the bib to the edge of the phantom in mm
            kk = 0  # counter for data points
            for profile in profiles:
                _, index = u.find_nearest(profile, 0.5)  # find the 50% amplitude point
                # value_near, index = find_nearest(profile, 0.5) # find the 50% amplitude point
                if (  # pylint: disable = consider-using-in
                    k == 0 or k == 2 
                ):  # there are the bibs in the horizontal
                    offset_value_y = round(
                        abs((ydet[k] - index) * (dy / 10)) - phantom_distance, 2
                    )

                    txt = str(offset_value_y)
                    # print('offset_value_y=', offset_value_y)
                    if abs(offset_value_y) <= tol:
                        Page.text(
                            0.1,
                            0.8 - kk / 10,
                            "Point" + str(kk + 1) + " offset=" + txt + " mm",
                            color="g",
                        )
                    elif abs(offset_value_y) > tol and abs(offset_value_y) <= act:
                        Page.text(
                            0.1,
                            0.8 - kk / 10,
                            "Point" + str(kk + 1) + " offset=" + txt + " mm",
                            color="y",
                        )
                    else:
                        Page.text(
                            0.1,
                            0.8 - kk / 10,
                            "Point" + str(kk + 1) + " offset=" + txt + " mm",
                            color="r",
                        )
                    kk = kk + 1

                    ax = fig.add_subplot(
                        4, 2, k + 1
                    )  # plotting all the figures in a single plot

                    ax.imshow(
                        np.array(imcirclist[k], dtype=np.uint8) / 255,
                        extent=list_extent[k],
                        origin="upper",
                    )
                    ax.scatter(
                        list_extent[k][0] + xdet[k] * dx / 100,
                        list_extent[k][3] + ydet[k] * dy / 100,
                        s=30,
                        marker="P",
                        color="y",
                    )
                    ax.set_title("Bib=" + str(k + 1))
                    ax.axhline(
                        list_extent[k][3] + index * dy / 100, color="r", linestyle="--"
                    )
                    ax.set_xlabel("x distance [cm]")
                    ax.set_ylabel("y distance [cm]")
                else:
                    offset_value_x = round(
                        abs((xdet[k] - index) * (dx / 10)) - phantom_distance, 2
                    )

                    txt = str(offset_value_x)
                    if abs(offset_value_x) <= tol:
                        # print('1')
                        Page.text(
                            0.1,
                            0.8 - kk / 10,
                            "Point" + str(kk + 1) + " offset=" + txt + " mm",
                            color="g",
                        )
                    elif abs(offset_value_x) > tol and abs(offset_value_x) <= act:
                        # print('2')
                        Page.text(
                            0.1,
                            0.8 - kk / 10,
                            "Point" + str(kk + 1) + " offset=" + txt + " mm",
                            color="y",
                        )
                    else:
                        # print('3')
                        Page.text(
                            0.1,
                            0.8 - kk / 10,
                            "Point" + str(kk + 1) + " offset=" + txt + " mm",
                            color="r",
                        )
                    kk = kk + 1

                    ax = fig.add_subplot(
                        4, 2, k + 1
                    )  # plotting all the figures in a single plot

                    ax.imshow(
                        np.array(imcirclist[k], dtype=np.uint8) / 255,
                        extent=list_extent[k],
                        origin="upper",
                    )
                    ax.scatter(
                        list_extent[k][0] + xdet[k] * dx / 100,
                        list_extent[k][3] + ydet[k] * dy / 100,
                        s=30,
                        marker="P",
                        color="y",
                    )
                    ax.set_title("Bib=" + str(k + 1))
                    ax.axvline(
                        list_extent[k][0] + index * dx / 100, color="r", linestyle="--"
                    )
                    ax.set_xlabel("x distance [cm]")
                    ax.set_ylabel("y distance [cm]")

                k = k + 1
        elif answers["type"] == "FC-2":
            phantom_distance = 7.0710678 # distance from the bib to the edge of the phantom in mm

            for kk in range(0,4):
                _, index_b = u.find_nearest(profiles[2*kk], 0.5)  # find the 50% amplitude point in the x direction
                _, index_a = u.find_nearest(profiles[2*kk+1], 0.5)  # find the 50% amplitude point in the y direction
                # value_near, index = find_nearest(profile, 0.5) # find the 50% amplitude point

                offset_value_x = round(abs((ydet[kk] - index_b) * (dy / 10)) - phantom_distance, 2)
                offset_value_y = round(abs((xdet[kk] - index_a) * (dx / 10)) - phantom_distance, 2)


                txt_x = str(offset_value_x)
                txt_y = str(offset_value_y)


                if abs(offset_value_y) <= tol:
                    Page.text(
                        0.1,
                        0.8 - 2*kk / 10,
                        "Point " + str(kk + 1) + " vertical offset=" + txt_y + " mm",
                        color="g",
                        )
                elif abs(offset_value_y) > tol and abs(offset_value_y) <= act:
                    Page.text(
                            0.1,
                            0.8 - 2*kk / 10,
                            "Point " + str(kk + 1) + " vertical offset=" + txt_y + " mm",
                            color="y",
                        )
                else:
                    Page.text(
                            0.1,
                            0.8 - 2*kk / 10,
                            "Point " + str(kk + 1) + " vertical offset=" + txt_y + " mm",
                            color="r",
                        )





                if abs(offset_value_x) <= tol:
                    Page.text(
                        0.1,
                        0.8 - (2*kk + 1) / 10,
                        "Point " + str(kk + 1) + " horizontal offset=" + txt_x + " mm",
                        color="g",
                    )
                elif abs(offset_value_x) > tol and abs(offset_value_x) <= act:
                    Page.text(
                        0.1,
                        0.8 - (2*kk + 1) / 10,
                        "Point " + str(kk + 1) + " horizontal offset=" + txt_x + " mm",
                        color="y",
                    )
                else:
                    Page.text(
                        0.1,
                        0.8 - (2*kk + 1) / 10,
                        "Point " + str(kk + 1) + " horizontal offset=" + txt_x + " mm",
                        color="r",
                    )






                    

                ax = fig.add_subplot(
                    4, 2, kk + 1
                )  # plotting all the figures in a single plot

                ax.imshow(
                    np.array(imcirclist[kk], dtype=np.uint8) / 255,
                    extent=list_extent[kk],
                    origin="upper",
                )
                ax.scatter(
                    list_extent[kk][0] + xdet[kk] * dx / 100,
                    list_extent[kk][3] + ydet[kk] * dy / 100,
                    s=30,
                    marker="P",
                    color="y",
                )
                ax.set_title("Bib=" + str(kk + 1))
                ax.axhline(
                    list_extent[kk][3] + index_b * dy / 100, color="r", linestyle="--"
                )

                ax.axvline(
                    list_extent[kk][0] + index_a * dx / 100, color="r", linestyle="--"
                )

                ax.set_xlabel("x distance [cm]")
                ax.set_ylabel("y distance [cm]")


        pdf.savefig()
        pdf.savefig(fig)

        # we now need to select a horizontal and a vertical profile to find the edge of the field from an image
        # for the field size calculation
        im = Image.fromarray(255 * ArrayDicom)

        if ioptn == 1:
            PROFILE = {
                "horizontal": 270,
                "vertical": 430,
            }  # location to extract the horizontal and vertical profiles if this is a linac
        else:
            PROFILE = {
                "horizontal": 470,
                "vertical": 510,
            }  # location to extract the horizontal and vertical profiles if this is a TrueBeam Edge of Varian XI

        profilehorz = (
            np.array(im, dtype=np.uint8)[PROFILE["horizontal"], :] / 255
        )  # we need to change these limits on a less specific criteria
        profilevert = np.array(im, dtype=np.uint8)[:, PROFILE["vertical"]] / 255

        _, index_top = u.find_nearest(
            profilevert[0 : height // 2], 0.5
        )  # finding the edge of the field on the top
        _, index_bot = u.find_nearest(
            profilevert[height // 2 : height], 0.5
        )  # finding the edge of the field on the bottom

        _, index_l = u.find_nearest(
            profilehorz[0 : width // 2], 0.5
        )  # finding the edge of the field on the bottom
        _, index_r = u.find_nearest(
            profilehorz[width // 2 : width], 0.5
        )  # finding the edge of the field on the right

        fig2 = plt.figure(
            figsize=(7, 5)
        )  # this figure will show the vertical and horizontal calculated field size
        ax = fig2.subplots()
        ax.imshow(ArrayDicom, extent=extent, origin="upper")
        ax.set_xlabel("x distance [cm]")
        ax.set_ylabel("y distance [cm]")

        # adding a vertical arrow
        ax.annotate(
            s="",
            xy=(PROFILE["vertical"] * dx / 10, index_top * dy / 10),
            xytext=(PROFILE["vertical"] * dx / 10, (height // 2 + index_bot) * dy / 10),
            arrowprops=dict(arrowstyle="<->", color="r"),
        )  # example on how to plot a double headed arrow
        ax.text(
            (PROFILE["vertical"] + 10) * dx / 10,
            (height // 2) * dy / 10,
            "Vfs="
            + str(round((height // 2 + index_bot - index_top) * dy / 10, 2))
            + "cm",
            rotation=90,
            fontsize=14,
            color="r",
        )

        # adding a horizontal arrow
        # print(index_l*dx, index_l, PROFILE['horizontal']*dy, PROFILE['horizontal'])
        ax.annotate(
            s="",
            xy=(index_l * dx / 10, PROFILE["horizontal"] * dy / 10),
            xytext=((width // 2 + index_r) * dx / 10, PROFILE["horizontal"] * dy / 10),
            arrowprops=dict(arrowstyle="<->", color="r"),
        )  # example on how to plot a double headed arrow
        ax.text(
            (width // 2) * dx / 10,
            (PROFILE["horizontal"] - 10) * dy / 10,
            "Hfs=" + str(round((width // 2 + index_r - index_l) * dx / 10, 2)) + "cm",
            rotation=0,
            fontsize=14,
            color="r",
        )

        pdf.savefig(fig2)


if __name__ == "__main__":
    # while True:  # example of infinite loops using try and except to catch only numbers
    #     line = input("Are these files from a clinac [yes(y)/no(n)]> ")
    #     try:
    #         ##        if line == 'done':
    #         ##            break
    #         ioption = str(line.lower())
    #         if ioption.startswith(("y", "yeah", "yes", "n", "no", "nope")):
    #             break
    #
    #     except:  # pylint: disable = bare-except
    #         print("Please enter a valid option:")

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Input the Light/Rad file")
    args = parser.parse_args()

    filename = args.file

    read_dicom(filename)
