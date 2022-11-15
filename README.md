# Invariant-TemplateMatching

Original post: https://forum.opencv.org/t/an-improved-template-matching-with-rotation-and-scale-invariant/

Template matching is a good approach for quick object detection but the template matching provided by OpenCV was not able to detect rotated and scaled in the match. Also it will generate many redundant matching boxes which is useless for robotic grasping. Therefore I made a few improvements on the original template matching. In my implementation, after cropping the template, I can set a range of rotate angles and scaling factors so the matching process will keep doing a grid search on all possible combinations of rotate angles and scaling factors. Also, I eliminate redundancies based on the size of the template. I packed functionalities into a new function and this made the robotic grasping based on template matching robust which will not be limited to different angles and sizes.

# Citing this code
Please cite the following paper:

Z. Zhang and H. Shang, "[Low-cost Solution for Vision-based Robotic Grasping](https://ieeexplore.ieee.org/document/9757984)," 2021 International Conference on Networking Systems of AI (INSAI), 2021, pp. 54-61, doi: 10.1109/INSAI54028.2021.00022.
