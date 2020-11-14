# Team Squeem

Collection of utility functions built by Team Squeem to be used in Python. 

### ClearDir
#### ex: ClearDir('./images')
Removes all files from a directory.
    
params:
- path - The source directory of the images to turn into a gif. Must include preceding ./, should not include ending /

### MakeGif
#### ex: MakeGif('./data', './', 'test', 100, 'jpg')
Turns a directory of images into a gif.
    
params:
- source_dir - The source directory of the images to turn into a gif. Must include preceding ./
- out_dir    - The directory to save the gif to. Must include preceding ./
- gif_name   - The name of the gif. Do not include filetype.
- duration   - Number of frames in the gif...I think.
- file_type  - File extension for the images. Do not include preceding .

### CreateMapBackground
#### ex: CreateMapBackground(edges=(-180,180,-90,90),buffer=0)
Edges = (Minimum Longitude, Maximum Longitude, Minimum Latitude, Maximum Latitude)
Buffer is the number of degrees between Min/Max Lon/Lat around the map