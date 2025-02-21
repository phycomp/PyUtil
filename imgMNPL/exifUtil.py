#!/usr/bin/env python
from argparse import ArgumentParser
from PIL.Image import open as imgOpen
from PIL.Image import FLIP_LEFT_RIGHT
from pexif.JpegFile import fromFile as jpgFile

def exifRot2(fname):
    img = JpgFile(fname)  #.fromFile
    try:
      orientation = img.exif.primary.Orientation[0]
      img.exif.primary.Orientation = [1]
      img.writeFile(temp_dir + filename)
    except: pass

def exifRot(fname):
  img = imgOpen(fname)
  if hasattr(img, '_getexif'):
      exif = img._getexif()
      for orientation in ExifTags.TAGS.keys():
          if ExifTags.TAGS[orientation]=='Orientation':
              break
      if exif and orientation in exif.keys():
          orientation = exif[orientation]
          if orientation is 6: img = img.rotate(-90, expand=True)
          elif orientation == 8: img = img.rotate(90, expand=True)
          elif orientation == 3: img = img.rotate(180, expand=True)
          elif orientation == 2: img = img.transpose(Image.FLIP_LEFT_RIGHT)
          elif orientation == 5: img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
          elif orientation == 7: img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
          elif orientation == 4: img = img.rotate(180, expand=True).transpose(Image.FLIP_LEFT_RIGHT)

#save the result
  img.save(temp_dir + filename)
if __name__=='__main__':
    parser = ArgumentParser(description='calculate stock to the total of SKY')
    parser.add_argument('--total', '-l', type=int, default=12000, help='the total stock')
    parser.add_argument('--sold', '-s', type=int, default=6000, help='the stock sold')
    parser.add_argument('--nmin', '-n', type=int, default=6, help='the minimal iterations')
    parser.add_argument('--max', '-m', type=float, default=.1, help='the maximum selling percentage')
    parser.add_argument('--Exif', '-E', action='store_true', default=True, help='Exif')
    args = parser.parse_args()
    if args.Calc: calc_stock(args)
    elif args.Iter: nStock(args)
    elif args.Exif:
        exifRot(args)
    elif args.Optimum: optimumPrice(args)
    elif args.Profit:
        sky=skyStock(args)
        sky.maxProfit()
