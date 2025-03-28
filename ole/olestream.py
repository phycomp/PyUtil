# -*- tab-width: 4; indent-tabs-mode: nil -*-
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

#import sys
#import pptGlobals
from pptGlobals import dumpBytes, ByteStream, indexbytes, outputln, encodeName, getUTF8FromUTF16, getRawBytes

class CompObjStreamError(Exception): pass

class MonikerStream(object):
    def __init__ (self, bytes):
        self.strm = ByteStream(bytes)

    def read (self):
        outputln("moniker size: %d"%(len(self.strm.bytes)-16))
        clsID = self.strm.readBytes(16)
        outputln("CLS ID: %s"%getRawBytes(clsID, True, False))
        outputln("stream data (implemention specific):")
        dumpBytes(self.strm.readRemainingBytes())
        outputln("")

class OLEStream(object):

    def __init__ (self, bytes):
        self.strm = ByteStream(bytes)

    def read (self):
        ver = self.strm.readUnsignedInt(4)
        outputln("version: 0x%8.8X"%ver)
        flags = self.strm.readUnsignedInt(4)
        outputln("flags: %d"%flags)
        linkUpdateOption = self.strm.readUnsignedInt(4)
        outputln("link update option: %d"%linkUpdateOption)
        reserved = self.strm.readUnsignedInt(4)
        outputln("")

        # Reserved moniker (must be ignored)
        monikerSize = self.strm.readUnsignedInt(4)
        if monikerSize > 0:
            strm = MonikerStream(self.strm.readBytes(monikerSize+16))
            strm.read()

        # Relative source moniker (relative path to the linked object)
        monikerSize = self.strm.readUnsignedInt(4)
        if monikerSize > 0:
            strm = MonikerStream(self.strm.readBytes(monikerSize+16))
            strm.read()

        # Absolute source moniker (absolute path to the linked object)
        monikerSize = self.strm.readUnsignedInt(4)
        if monikerSize > 0:
            strm = MonikerStream(self.strm.readBytes(monikerSize+16))
            strm.read()

        clsIDIndicator = self.strm.readSignedInt(4)
        outputln("cls ID indicator: %d"%clsIDIndicator)
        clsID = self.strm.readBytes(16)
        outputln("CLS ID: %s"%getRawBytes(clsID, True, False))
#       dumpBytes(self.strm.bytes, 512)

class CompObjStream(object):

    def __init__ (self, bytes):
        self.strm = ByteStream(bytes)

    def read (self):
        # CompObjHeader
        reserved = self.strm.readBytes(4)
        ver = self.strm.readUnsignedInt(4)
        reserved = self.strm.readBytes(20)
        outputln("version: 0x%4.4X"%ver)

        # LengthPrefixedAnsiString
        length = self.strm.readUnsignedInt(4)
        displayName = self.strm.readBytes(length)
        if indexbytes(displayName, -1) != 0x00:
            # must be null-terminated.
            raise CompObjStreamError()

        outputln("display name: " + encodeName(displayName[:-1]))

        # ClipboardFormatOrAnsiString
        marker = self.strm.readUnsignedInt(4)
        if marker == 0:
            # Don't do anything.
            pass
        elif marker == 0xFFFFFFFF or marker == 0xFFFFFFFE:
            clipFormatID = self.strm.readUnsignedInt(4)
            outputln("clipboard format ID: %d"%clipFormatID)
        else:
            clipName = self.strm.readBytes(marker)
            if indexbytes(clipName, -1) != 0x00:
                # must be null-terminated.
                raise CompObjStreamError()
            outputln("clipboard format name: %s"%clipName[:-1])

        # LengthPrefixedAnsiString
        length = self.strm.readUnsignedInt(4)
        if length == 0 or length > 0x00000028:
            # the spec says this stream is now invalid.
            raise CompObjStreamError()

        reserved = self.strm.readBytes(length)
        if indexbytes(reserved, -1) != 0x00:
            # must be null-terminated.
            raise CompObjStreamError()

        outputln("reserved name : %s"%reserved[:-1])
        unicodeMarker = self.strm.readUnsignedInt(4)
        if unicodeMarker != 0x71B239F4:
            raise CompObjStreamError()

        # LengthPrefixedUnicodeString
        length = self.strm.readUnsignedInt(4)
        if length > 0:
            s = getUTF8FromUTF16(self.strm.readBytes(length*2))
            outputln("display name (unicode): %s"%s)

        # ClipboardFormatOrAnsiString
        marker = self.strm.readUnsignedInt(4)
        if marker == 0:
            # Don't do anything.
            pass
        elif marker == 0xFFFFFFFF or marker == 0xFFFFFFFE:
            clipFormatID = self.strm.readUnsignedInt(4)
            outputln("clipboard format ID: %d"%clipFormatID)
        else:
            clipName = getUTF8FromUTF16(self.strm.readBytes(marker*2))
            if indexbytes(clipName, -1) != 0x00:
                # must be null-terminated.
                raise CompObjStreamError()
            outputln("clipboard format name: %s"%clipName[:-1])

class PropertySetStream(object):

    def __init__ (self, bytes):
        self.strm = ByteStream(bytes)

    def read (self):
        byteorder = self.strm.readUnsignedInt(2)
        if byteorder != 0xFFFE:
            raise PropertySetStreamError()
        ver = self.strm.readUnsignedInt(2)
        outputln("version: 0x%4.4X"%ver)
        sID = self.strm.readUnsignedInt(4)
        outputln("system identifier: 0x%4.4X"%sID)
        clsID = self.strm.readBytes(16)
        outputln("CLS ID: %s"%getRawBytes(clsID, True, False))
        sets = self.strm.readUnsignedInt(4)
        outputln("number of property sets: 0x%4.4X"%sets)
        fmtID0 = self.strm.readBytes(16)
        outputln("FMT ID 0: %s"%getRawBytes(fmtID0, True, False))
        offset0 = self.strm.readUnsignedInt(4)
        outputln("offset 0: 0x%4.4X"%offset0)
        if sets > 1:
            fmtID1 = self.strm.readBytes(16)
            outputln("FMT ID 1: %s"%getRawBytes(fmtID0, True, False))
            offset1 = self.strm.readUnsignedInt(4)
            outputln("offset 1: 0x%4.4X\n"%offset1)
        self.readSet(offset0)
        if sets > 1:
            self.strm.setCurrentPos(offset1);
            self.readSet(offset1)

    def readSet (self, setOffset):
        outputln("-----------------------------")
        outputln("Property set")
        outputln("-----------------------------")
        size = self.strm.readUnsignedInt(4)
        outputln("size: 0x%4.4X"%size)
        props = self.strm.readUnsignedInt(4)
        outputln("number of properties: 0x%4.4X"%props)
        pos = 0
        while pos < props:
            self.strm.setCurrentPos(setOffset + 8 + pos*8);
            id = self.strm.readUnsignedInt(4)
            offset = self.strm.readUnsignedInt(4)
            outputln("ID: 0x%4.4X offset: 0x%4.4X"%(id, offset))
            self.strm.setCurrentPos(setOffset + offset);
            type = self.strm.readUnsignedInt(2)
            padding = self.strm.readUnsignedInt(2)
            if padding != 0:
                raise PropertySetStreamError()
            outputln("type: 0x%4.4X"%type)
            if type == 2:
                value = self.strm.readSignedInt(2)
                outputln("VT_I2: %d"%value)
            elif type == 0x41:
                blobSize = self.strm.readUnsignedInt(4)
                outputln("VT_BLOB size: 0x%4.4X"%blobSize)
                outputln("------------------------------------------------------------------------")
                dumpBytes(self.strm.bytes[self.strm.pos:self.strm.pos+blobSize], blobSize)
                outputln("------------------------------------------------------------------------")
            else:
                outputln("unknown type")
            pos += 1
        outputln("")

# vim:set filetype=python shiftwidth=4 softtabstop=4 expandtab:
