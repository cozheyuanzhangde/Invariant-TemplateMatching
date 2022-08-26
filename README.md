# Invariant-TemplateMatching

Original Post: https://forum.opencv.org/t/an-improved-template-matching-with-rotation-and-scale-invariant/

Template matching is a good approach for quick object localization but the template matching provided by OpenCV was not able to detect rotated and scaled in the match. Also it will generate many redundant matching boxes which is useless for robotic grasping. Therefore I made a few improvements on the original template matching. In my implementation, after cropping the template, I can set a range of rotate angles and scaling factors so the matching process will keep do a grid search on all possible combinations of rotate angles and scaling factors. Also, I eliminate redundancies based on the size of the template. I packed some of the new functionalities into a new function and this made the robotic grasping based on template matching robust which will not be limited to different angles and sizes.
