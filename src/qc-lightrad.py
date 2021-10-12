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
#   aims to dynamically identify them using machine vision and the BBs in the phantom.
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
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from PIL import Image
from skimage.feature import blob_log
import pydicom
import roi_sel as roi
import utils as u
import inquirer
from timeit import default_timer as timer
from outlier import find_outlier_pixels


def argmax2d(X):
    n, m = X.shape
    x_ = np.ravel(X)
    k = np.argmax(x_)
    i, j = k // m, k % m
    return i, j


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
        -(ArrayDicom.shape[1] / 2) * dx / 10,
        (ArrayDicom.shape[1] / 2) * dx / 10,
        -(ArrayDicom.shape[0] / 2) * dy / 10,
        (ArrayDicom.shape[0] / 2) * dy / 10,
    )

    # creating the figure extent list for the BB images
    list_extent = []

    print('np.shape_dim0',np.shape(ArrayDicom)[0])
    if np.shape(ArrayDicom)[0] == 768: #Clinac
        ioptn=1
    elif np.shape(ArrayDicom)[0] == 1190: #Edge EPID
        ioptn=2
    elif np.shape(ArrayDicom)[0] == 1280: ##Varian XI (Edmonton)
        ioptn=3

    intercept = dataset[0x0028, 0x1052].value
    slope = dataset[0x0028, 0x1053].value
    ArrayDicom = ArrayDicom * slope + intercept



    if ioptn == 1:
        height, width = ArrayDicom.shape
        ArrayDicom_mod = ArrayDicom[:, width // 2 - height // 2 : width // 2 + height // 2]
    else:
        ArrayDicom_mod = ArrayDicom

    # we take a diagonal profile to avoid phantom artifacts
    # im_profile = ArrayDicom_mod.diagonal()

    # test to make sure image is displayed correctly BBs are high amplitude against dark background
    ctr_pixel = ArrayDicom_mod[height // 2, width // 2]
    corner_pixel = ArrayDicom_mod[0, 0]


    # processing of hot BBs
    hot_pixels, ArrayDicom = find_outlier_pixels(ArrayDicom)



    ArrayDicom = u.norm01(ArrayDicom)

    if ctr_pixel > corner_pixel:
        ArrayDicom = u.range_invert(ArrayDicom)




    #Here we need to select the correct ROIs for the image processing since in the FC-2 phantom the BBs are in the corners we will select more than 1 ROI per-BB maybe??
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
    print('ioptn',ioptn)

    if answers["type"] == "IsoAlign":
        profiles, imcirclist, xdet, ydet, list_extent = roi.roi_sel_IsoAlign(ArrayDicom,ioptn,dx,dy)
    elif answers["type"] == "FC-2":
        profiles, imcirclist, xdet, ydet, list_extent = roi.roi_sel_FC2(ArrayDicom,ioptn,dx,dy)
    elif answers["type"] == "GP-1":
        start = timer()
        profiles, imcirclist, point, list_extent = roi.roi_sel_GP1(ArrayDicom,ioptn,dx,dy)
        dt = timer() - start
        print("File processed in %f s" % dt)



    
    # tolerance levels to change at will
    tol = 1.0  # tolearance level
    act = 2.0  # action level



    k = 0
    fig = plt.figure(figsize=(8, 12))  # this figure will hold the BBs
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
            phantom_distance = 3.0  # distance from the BB to the edge of the phantom in mm
            kk = 0  # counter for data points
            for profile in profiles:
                profile_inv = u.range_invert(profile)
                _, index = u.find_nearest(profile_inv, 0.5)  # find the 50% amplitude point
                # value_near, index = find_nearest(profile, 0.5) # find the 50% amplitude point
                
                if (  # pylint: disable = consider-using-in
                    k == 0 or k == 1 or k == 4 or k == 5
                ):  # there are the BBs in the horizontal
                    offset_value_y = round(
                        abs((point[k][1] - index) * (dy / 10)) - phantom_distance, 2
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
                        list_extent[k][0] + point[k][0] * dx / 100,
                        list_extent[k][3] + point[k][0] * dy / 100,
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
                        abs((point[k][0] - index) * (dx / 10)) - phantom_distance, 2
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
                        list_extent[k][0] + point[k][0] * dx / 100,
                        list_extent[k][3] + point[k][1] * dy / 100,
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
            phantom_distance = 10.0  # distance from the BB to the edge of the phantom in mm
            kk = 0  # counter for data points
            for profile in profiles:
                profile_inv = u.range_invert(profile)
                _, index = u.find_nearest(profile_inv, 0.5)  # find the 50% amplitude point
                # value_near, index = find_nearest(profile, 0.5) # find the 50% amplitude point
                if (  # pylint: disable = consider-using-in
                    k == 0 or k == 2 
                ):  # there are the BBs in the horizontal
                    offset_value_y = round(
                        abs((point[k][1] - index) * (dy / 10)) - phantom_distance, 2
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
                        list_extent[k][0] + point[k][0] * dx / 100,
                        list_extent[k][3] + point[k][1] * dy / 100,
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
                        abs((point[k][0] - index) * (dx / 10)) - phantom_distance, 2
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
                        list_extent[k][0] + point[k][0] * dx / 100,
                        list_extent[k][3] + point[k][1] * dy / 100,
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
            phantom_distance = 7.0710678 # distance from the BB to the edge of the phantom in mm

            for kk in range(0,4):
                profile_inv1 = u.range_invert(profiles[2*kk])
                profile_inv2 = u.range_invert(profiles[2*kk+1])
                _, index_b = u.find_nearest(profile_inv1, 0.5)  # find the 50% amplitude point in the x direction
                _, index_a = u.find_nearest(profile_inv2, 0.5)  # find the 50% amplitude point in the y direction
                # value_near, index = find_nearest(profile, 0.5) # find the 50% amplitude point

                offset_value_y = round(abs((point[kk][1] - index_b) * (dy / 10)) - phantom_distance, 2)
                offset_value_x = round(abs((point[kk][0] - index_a) * (dx / 10)) - phantom_distance, 2)


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
                    list_extent[kk][0] + point[kk][0] * dx / 100,
                    list_extent[kk][3] + point[kk][1] * dy / 100,
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
        elif ioptn == 2:
            PROFILE = {
                "horizontal": 470,
                "vertical": 510,
            }  # location to extract the horizontal and vertical profiles if this is a TrueBeam Edge of Varian XI
        elif ioptn == 3:
            PROFILE = {
                "horizontal": 470,
                "vertical": 470,
            }  # location to extract the horizontal and vertical profiles if this is a TrueBeam Edge of Varian XI
        

        #new profile definitions
        profilehorz = u.range_invert(ArrayDicom[height//2,:])
        profilevert = u.range_invert(ArrayDicom[:, width//2])
        print(np.shape(profilehorz),np.shape(profilevert))

        profilehorz_im = Image.fromarray(255*profilehorz)
        profilehorz_interp = profilehorz_im.resize((1,  profilehorz_im.height*10), Image.LANCZOS)

        profilevert_im = Image.fromarray(255*profilevert)
        profilevert_interp = profilevert_im.resize((1,  profilevert_im.height*10), Image.LANCZOS)

        profilehorz = np.array(profilehorz_interp)/255
        profilevert = np.array(profilevert_interp)/255

        _, index_top = u.find_nearest(
        profilevert[0 : height*10 // 2], 0.5
        )  # finding the edge of the field on the top
        _, index_bot = u.find_nearest(
            profilevert[height*10 // 2 : height*10], 0.5
        )  # finding the edge of the field on the bottom

        _, index_l = u.find_nearest(
            profilehorz[0 : width*10 // 2], 0.5
        )  # finding the edge of the field on the left
        _, index_r = u.find_nearest(
            profilehorz[width*10 // 2 : width*10], 0.5
        )  # finding the edge of the field on the right



        fig2 = plt.figure(
            figsize=(7, 5)
        )  # this figure will show the vertical and horizontal calculated field size
        ax = fig2.subplots()
        ax.imshow(ArrayDicom, extent=extent, origin="upper",    cmap="jet_r")
        ax.set_xlabel("x distance [cm]")
        ax.set_ylabel("y distance [cm]")
        ax.set_xlim(-7.5, 7.5)
        ax.set_ylim(-7.5, 7.5)

        # Change tick spacing
        ax.xaxis.set_major_locator(MultipleLocator(2.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))

        ax.yaxis.set_major_locator(MultipleLocator(2.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))

        # adding a vertical arrow
        Vert_Field_Size = round((height*10 // 2 + index_bot - index_top)//10 * dy / 10, 2)
        ax.annotate(
        text="",
        xy=(
            (0) * dx / 10,
            (height*10 // 2 - index_top)//10 * dy / 10,
        ),
        xytext=(
            (0) * dx / 10,
            -(index_bot)//10 * dy / 10,
        ),
        arrowprops=dict(arrowstyle="<->", color="lime", linewidth=1.5),
        )  # example on how to plot a double headed arrow
        ax.text(
        -(width // 2) * 0.45 * dx / 10,
        0,
        f"Vertical Field Size\n{Vert_Field_Size:.2f} cm",
        rotation=90,
        fontsize=10,
        color="lime",
        verticalalignment="center",
        ha="center",
        )

        # adding a horizontal arrow
        Hor_Field_Size = round((width*10 // 2 + index_r - index_l)//10 * dx / 10, 2)
        ax.annotate(
        text="",
        xy=(
            (index_r)//10 * dx / 10,
            (0) * dy / 10,
        ),
        xytext=(
            (index_l - (width*10 / 2))//10 * dx / 10,
            (0) * dy / 10,
        ),
        arrowprops=dict(arrowstyle="<->", color="cyan", linewidth=1.5),
        )  # example on how to plot a double headed arrow
        ax.text(
        0,
        (height // 2) * 0.4 * dy / 10,
        f"Horizontal Field Size\n{Hor_Field_Size:.2f} cm",
        rotation=0,
        fontsize=10,
        color="cyan",
        ha="center",
        )


        print('index_l=',index_l, 'index_r=',index_r)
        print('height=',height,'width=',width, 'height*10/2=',height*10/2, 'width*10/2=',width*10/2)
        plt.figure()
        plt.scatter(np.arange(len(profilehorz[0 : width*10 // 2])),profilehorz[0 : width*10 // 2])
        plt.figure()
        plt.scatter(np.arange(len(profilehorz[width*10 // 2 : width*10])),profilehorz[width*10 // 2 : width*10])
        plt.show()



        print('index_l=',index_l, 'index_r=',index_r)
        print('height=',height,'width=',width, 'height*10/2=',height*10/2, 'width*10/2=',width*10/2)
        plt.figure()
        plt.scatter(np.arange(len(profilehorz[0 : width*10 // 2])),profilehorz[0 : width*10 // 2])
        plt.figure()
        plt.scatter(np.arange(len(profilehorz[width*10 // 2 : width*10])),profilehorz[width*10 // 2 : width*10])
        plt.show()

        # Adding the field edges on the image
        ax.axhline(y=(height*10/2-index_top)* dy /100, linestyle="dashed")
        ax.axhline(y=(-index_bot) * dy /100, linestyle="dashed")
        ax.axvline(x=(width*10/2-index_l)/10 * dx /10, linestyle="dashed")
        ax.axvline(x=(-index_r)/10 * dx /10, linestyle="dashed")



        pdf.savefig(fig2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Input the Light/Rad file")
    args = parser.parse_args()

    filename = args.file

    read_dicom(filename)
