from zipfile import ZipFile
import os
from os.path import basename
# create a ZipFile object
with ZipFile('sampleDir.zip', 'w') as zipObj:
    # Iterate over all the files in directory
    for folderName, subfolders, filenames in os.walk("static\images"):
        for filename in filenames:
            # create complete filepath of file in directory
            filePath = os.path.join(folderName, filename)
            # Add file to zip
            zipObj.write(filePath, basename(filePath))
