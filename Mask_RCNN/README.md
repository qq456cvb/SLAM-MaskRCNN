`from dmask import mask_detect`

mask_detect(model, rgb_image, depth_image=None)

model: pretrained mask RCNN detection model

rgb_image: rgb input

depth_image(optional): get rid of unreasonable region if have realiable depth input
