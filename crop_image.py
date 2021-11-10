import glob
import os
import shutil
import cv2

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized

def main():
    fold_home = '/home/hust/source/motpred/data/201904/short/'
    # action_set = ["walking", "eating", "smoking", "discussion"]
    action_set = os.listdir(fold_home)

    for action in action_set:
        fold = '%s/%s' % (fold_home, action)
        img_lst = glob.glob('%s/*.jpg' % fold)
        # if not os.path.exists(fold + '/sub'):
        #     os.makedirs(fold + '/sub')

        for img_name in img_lst:
            if 'red_frm_' in img_name:
                # fold_path, name = os.path.split(img_name)
                # print(os.path.join(fold_path, 'ours' + name[3:]) )
                # shutil.move(img_name, os.path.join(fold_path, 'ours' + name[3:]) )
                pass
            # crop_image(img_name)
            # if int(img_name.split('.')[0].split('_')[-1]) in range(52, 50+25*4, 4):
            #     shutil.move(img_name, fold + '/sub')

            if 0:
                image = cv2.imread(img_name)

                # if image.shape[0] > 600:
                #     resized_image = cv2.resize(image, (200, 600))
                #     cv2.imwrite(img_name, resized_image)

                # resized_image = cv2.resize(image, (900, 2000))
                # resized_image = image_resize(image, width=200)
                # cv2.imwrite(img_name, resized_image)

def crop_image(img_name):
    '''
    automatically remove image margin
    '''
    # load the image
    image = cv2.imread(img_name)
    margin = 2
    gray_image = 255 - cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, bn_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    im2, contours, hierarchy = cv2.findContours(bn_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # find the biggest area of the contour
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)
    # draw the 'human' contour (in green)
    x1, y1 = x - margin, y - margin
    x2, y2 = x + w + margin, y + h + margin
    # crop_img = cv2.rectangle(output,(x1,y1),(x2, y2),(0,255,0),2)
    crop_img = image[y:y + h, x:x + w]
    cv2.imwrite(img_name, crop_img)
    # # show the image
    # cv2.imshow("Result", output)
    # cv2.waitKey(0)
    return 0

if __name__ == '__main__':
    main()