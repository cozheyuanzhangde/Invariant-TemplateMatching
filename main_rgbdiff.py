import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

from InvariantTM import template_crop, invariant_match_template

if __name__ == "__main__":
    img_bgr = cv2.imread('./images/image_2.jpg')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    template_bgr = plt.imread('./images/template_2.jpg')
    template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)
    cropped_template_rgb = template_crop(template_rgb)
    cropped_template_rgb = np.array(cropped_template_rgb)
    cropped_template_gray = cv2.cvtColor(cropped_template_rgb, cv2.COLOR_RGB2GRAY)
    height, width = cropped_template_gray.shape
    fig = plt.figure(num='Template - Close the Window to Continue >>>')
    plt.imshow(cropped_template_rgb)
    plt.show()
    points_list = invariant_match_template(rgbimage=img_rgb, rgbtemplate=cropped_template_rgb, method="TM_CCOEFF_NORMED", matched_thresh=0.5, rot_range=[0,360], rot_interval=10, scale_range=[100,150], scale_interval=10, rm_redundant=True, minmax=True, rgbdiff_thresh=215.0)
    fig, ax = plt.subplots(1)
    plt.gcf().canvas.manager.set_window_title('Template Matching Results: Rectangles')
    ax.imshow(img_rgb)
    centers_list = []
    for point_info in points_list:
        point = point_info[0]
        angle = point_info[1]
        scale = point_info[2]
        print(f"matched point: {point}, angle: {angle}, scale: {scale}")
        centers_list.append([point, scale])
        plt.scatter(point[0] + (width/2)*scale/100, point[1] + (height/2)*scale/100, s=20, color="red")
        plt.scatter(point[0], point[1], s=20, color="green")
        rectangle = patches.Rectangle((point[0], point[1]), width*scale/100, height*scale/100, color="red", alpha=0.50, label='Matched box')
        box = patches.Rectangle((point[0], point[1]), width*scale/100, height*scale/100, color="green", alpha=0.50, label='Bounding box')
        transform = mpl.transforms.Affine2D().rotate_deg_around(point[0] + width/2*scale/100, point[1] + height/2*scale/100, angle) + ax.transData
        rectangle.set_transform(transform)
        ax.add_patch(rectangle)
        ax.add_patch(box)
        plt.legend(handles=[rectangle,box])
    #plt.grid(True)
    plt.show()
    fig2, ax2 = plt.subplots(1)
    plt.gcf().canvas.manager.set_window_title('Template Matching Results: Centers')
    ax2.imshow(img_rgb)
    for point_info in centers_list:
        point = point_info[0]
        scale = point_info[1]
        plt.scatter(point[0]+width/2*scale/100, point[1]+height/2*scale/100, s=20, color="red")
    plt.show()