from PIL import Image
from PIL.ExifTags import TAGS 
import imghdr

def main():
    # path to the image or video
    img_path = './tower.png'
    print(img_path)
    print(imghdr.what(img_path))

    # read the image data using PIL
    image = Image.open(img_path)
    # print(image)

    # extract other basic metadata
    info_dict = {
        "Filename" : image.filename,
        "Image Size" : image.size,
        "Image Height" : image.height,
        "Image Width" : image.width,
        "Image Format" : image.format,
        "Image Mode" : image.mode,
        "Animated Image" : getattr(image, "is_animated", False),
        "Frames in Image" : getattr(image, "n_frames", 1)
    }

    # showing basic information
    for label, value in info_dict.items():
        print(f"{label:25}: {value}")
    
    # extract EXIF data
    exifdata = image.getexif()

    # iterating over all EXIF data fields
    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tad id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes
        if isinstance(data, bytes):
            data = data.decode()
        print(f"{tag:25}: {data}")
    
    # jpg to png conversion
    image2 = image.convert('RGB')
    image2.save("tower2.jpeg")

if __name__ == '__main__':
    main()