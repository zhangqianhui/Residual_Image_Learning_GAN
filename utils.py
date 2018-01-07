import os
import errno
import numpy as np
import scipy
import scipy.misc

def mkdir_p(path):

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_image(image_path , image_size , is_crop=True, resize_w=64 , is_grayscale = False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w)

def transform(image, npx=64 , is_crop=False, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w = resize_w)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])
    return np.array(cropped_image)/127.5 - 1

def center_crop(x, crop_h, crop_w=None, resize_w=64):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    rate = np.random.uniform(0, 1, size=1)

    if rate < 0.5:
        x = np.fliplr(x)

    #first crop tp 178x178 and resize to 128x128
    return scipy.misc.imresize(x[20:218-20, 0: 178], [resize_w, resize_w])

    #Another cropped method

    # return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
    #                            [resize_w, resize_w])

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale=False):

    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):

    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image):
    return ((image + 1)* 127.5).astype(np.uint8)

def read_image_list(category):

    filenames = []
    print("list file")
    list = os.listdir(category)
    list.sort()
    for file in list:
        if 'jpg' or 'png' in file:
            filenames.append(category + "/" + file)
    print("list file ending!")

    length = len(filenames)
    perm = np.arange(length)
    np.random.shuffle(perm)
    filenames = np.array(filenames)
    filenames = filenames[perm]

    return filenames

class CelebA(object):

    def __init__(self, images_path, image_size):

        self.dataname = "CelebA"
        self.dims = image_size*image_size
        self.shape = [image_size, image_size, 3]
        self.image_size = image_size
        self.channel = 3
        self.images_path = images_path
        self.dom_1_train_data_list, self.dom_1_train_lab_list, self.dom_2_train_data_list, self.dom_2_train_lab_list = self.load_celebA()

        self.train_len = 0

        if len(self.dom_1_train_data_list) > len(self.dom_2_train_data_list):

            self.train_len = len(self.dom_2_train_data_list)
        else:
            self.train_len = len(self.dom_1_train_data_list)

    def load_celebA(self):

        # get the list of image path
        return read_image_list_file(self.images_path, is_test=False)

    def load_test_celebA(self):

        # get the list of image path
        return read_image_list_file(self.images_path, is_test=True)

    def getShapeForData(self, filenames):

        array = [get_image(batch_file, 128, is_crop=True, resize_w=self.image_size,
                           is_grayscale=False) for batch_file in filenames]
        sample_images = np.array(array)

        return sample_images

    def getNextBatch(self, batch_num=0, batch_size=64):

        ro_num = self.train_len / 64

        if batch_num % ro_num == 0:

            perm = np.arange(self.train_len)

            np.random.shuffle(perm)

            self.dom_1_train_data_list = np.array(self.dom_1_train_data_list)
            self.dom_1_train_data_list = self.dom_1_train_data_list[perm]
            self.dom_2_train_data_list = np.array(self.dom_2_train_data_list)
            self.dom_2_train_data_list = self.dom_2_train_data_list[perm]

            self.dom_1_train_lab_list = np.array(self.dom_1_train_lab_list)
            self.dom_1_train_lab_list = self.dom_1_train_lab_list[perm]

            self.dom_2_train_lab_list = np.array(self.dom_2_train_lab_list)
            self.dom_2_train_lab_list = self.dom_2_train_lab_list[perm]

        return self.dom_1_train_data_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.dom_1_train_lab_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.dom_2_train_data_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.dom_2_train_lab_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]

    def getTestNextBatch(self, batch_num=0, batch_size=64):

        ro_num = len(self.test_data_list) / batch_size
        if batch_num % ro_num == 0:

            length = len(self.test_data_list)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.test_data_list = np.array(self.test_data_list)
            self.test_data_list = self.test_data_list[perm]
            self.test_lab_list = np.array(self.test_lab_list)
            self.test_lab_list = self.test_lab_list[perm]

        return self.test_data_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_lab_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]

def read_image_list_file(category, is_test):

    end_num = 0
    start_num = 1202

    dom_1_list_image = []
    dom_1_list_label = []

    dom_2_list_image = []
    dom_2_list_label = []

    lines = open(category + "../" + "list_attr_celeba.txt")
    li_num = 0
    for line in lines:

        if li_num < start_num:
            li_num += 1
            continue

        if li_num >= end_num and is_test == True:
            break

        flag = line.split('1 ', 41)[20]  # get the label for gender
        file_name = line.split(' ', 1)[0]

        # print flag
        if flag == ' ':

            dom_1_list_image.append(category + file_name)
            dom_1_list_label.append(1)
            
        else:
            
            dom_2_list_image.append(category + file_name)
            dom_2_list_label.append(0)

        li_num += 1

    lines.close()
    #keep the balance of the dataset.
    return dom_1_list_image[0:80000], dom_1_list_label[0:80000], dom_2_list_image[0:80000], dom_2_list_label[0:80000]



