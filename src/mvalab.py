# Lecture des fichiers "télécom"
# Sylvain Lobry 2015
# JM Nicolas 2015

"""
lecture, écriture et affichage d'images à forte dynamique, réeles ou complexes (radar)
"""

MVALABVERSION = "V2.1  Version du 5 février 2019"

import numpy as npy
import struct

globalparamnotebook = 0

typecode = "<f"  # < : little endian
hdrcode = "byte order = 0"  # pour .hdr d'IDL/ENVI
imacode = "-byteorder = 0"  # pour .dim (mesure conservatoire)


def mat2imz(tabimage, nomimage, *therest):
    """
    Procedure pour ecrire un tableau dans un fichier au format TelecomParisTech
    Le tableau sera archivé en :
        .ima si tableau 8 bits
        .IMF sinon
        .CXF si complexe
    Si le tableau est à 3 dimensions (pile TIVOLI), l'archivage se fera en .IMA
    Exemple d'appel :
    mat2imz( montableau2d, 'MaSortie')
    Pour avoir aussi  le fichier .hdr d'IDL
    mat2imz( montableau2d, 'MaSortie', 'idl')
    """

    nomdim = nomimage + ".dim"

    taghdr = 0
    testchar = 0

    if len(therest) == 1:
        if therest[0] == "idl":
            taghdr = 1

    ndim = npy.ndim(tabimage)
    if ndim < 2:
        print("mat2imz demande un tableau 2D ou 3D")
        return
    if ndim > 3:
        print("mat2imz demande un tableau 2D ou 3D")
        return

    nlig = npy.size(tabimage, 0)
    ncol = npy.size(tabimage, 1)
    nplan = 1  # par defaut.. pour idl
    #
    # Cas image 2D
    #
    if ndim == 2:
        fp = open(nomdim, "w")
        fp.write("%d" % ncol + "  %d" % nlig)
        fp.close()
        imode = npy.iscomplex(tabimage[0][0])
        if imode == True:
            nomimagetot = nomimage + ".CXF"
            fp = open(nomimagetot, "wb")
            for iut in range(nlig):
                for jut in range(ncol):
                    fbuff = float(tabimage.real[iut][jut])
                    record = struct.pack(typecode, fbuff)
                    fp.write(record)
                    fbuff = float(tabimage.imag[iut][jut])
                    record = struct.pack(typecode, fbuff)
                    fp.write(record)
            fp.close()

        else:
            mintab = npy.min(tabimage)
            maxtab = npy.max(tabimage)
            if mintab > -0.0001:
                if maxtab < 255.0001:
                    testchar = 1
                    nomimagetot = nomimage + ".ima"
                    ucima = npy.uint8(tabimage)
                    fp = open(nomimagetot, "wb")
                    for iut in range(nlig):
                        for jut in range(ncol):
                            record = struct.pack("B", ucima[iut][jut])
                            fp.write(record)

                else:
                    nomimagetot = nomimage + ".IMF"
                    fp = open(nomimagetot, "wb")
                    for iut in range(nlig):
                        for jut in range(ncol):
                            fbuff = float(tabimage[iut][jut])
                            record = struct.pack(typecode, fbuff)
                            fp.write(record)
            fp.close()

    if ndim == 3:
        nplan = npy.size(tabimage, 2)
        imode = npy.iscomplex(tabimage[0][0][0])
        mintab = npy.min(tabimage)
        maxtab = npy.max(tabimage)
        if mintab > -0.0001:
            if maxtab < 255.0001:
                testchar = 1

        fp = open(nomdim, "w")
        fp.write("%d" % ncol + "  %d" % nlig + "  %d" % nplan + "   1" + "\n")
        if imode == True:
            fp.write("-type CFLOAT")
        else:
            if testchar == 0:
                fp.write("-type FLOAT")
            if testchar == 1:
                fp.write("-type U8")
        fp.close()

        nomimagetot = nomimage + ".IMA"
        fp = open(nomimagetot, "wb")
        if imode == True:
            for lut in range(nplan):
                for iut in range(nlig):
                    for jut in range(ncol):
                        fbuff = float(tabimage.real[iut][jut][lut])
                        record = struct.pack(typecode, fbuff)
                        fp.write(record)
                        fbuff = float(tabimage.imag[iut][jut][lut])
                        record = struct.pack(typecode, fbuff)
                        fp.write(record)
        else:
            if testchar == 1:
                for lut in range(nplan):
                    ucima = npy.uint8(tabimage[:, :, lut])
                    for iut in range(nlig):
                        for jut in range(ncol):
                            record = struct.pack("B", ucima[iut][jut])
                            fp.write(record)
            else:
                for lut in range(nplan):
                    for iut in range(nlig):
                        for jut in range(ncol):
                            fbuff = float(tabimage[iut][jut][lut])
                            record = struct.pack(typecode, fbuff)
                            fp.write(record)

        fp.close()

    if taghdr == 1:
        noffset = 0
        nomhdr = nomimagetot + ".hdr"
        fp = open(nomhdr, "w")
        fp.write("ENVI \n")
        fp.write("{Fichier produit par tiilab.mat2imz (python) } \n")
        fp.write("samples = %d" % ncol + "\n")
        fp.write("lines = %d" % nlig + "\n")
        fp.write("bands = %d" % nplan + "\n")
        fp.write("header offset = %d" % noffset + "\n")
        fp.write("file type = ENVI Standard \n")
        if imode == True:
            fp.write("data type = 6 \n")
        else:
            if testchar == 1:
                fp.write("data type = 1  \n")
            else:
                fp.write("data type = 4  \n")

        fp.write("interleave = bsq \n")
        fp.write(hdrcode + "\n")
        if imode == True:
            fp.write("complex function = Magnitude  \n")

        fp.close()
