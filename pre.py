import os  
from tqdm import tqdm
import cv2

TRAIN_DIR = './train'
IMG_SIZE = 50
'''desired_size = 368
im_pth = "/train.jpg"
im = cv2.imread(im_pth)
old_size = im.shape[:2]
ratio = float(desired_size)/max(old_size)
new_size= tuple([int(x*ratio) for x in old_size])
im = cv2.resize(im, (new_size[1], new_size[0]))
delta_w= desired_size - new_size[1]
delta_h = desired_size - new_size[0]
top, bottom = delta_h//2, delta_h-(delta_h//2)
left, right = delta_w//2, delta_w-(delta_w//2)

color = [0,0,0]
new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
cv2.imshow("image", new_im)
cv2.waitKey(0)
cv2.destroyAllWindows()'''


def create_train_test_dir():
    counter=1
    for img in tqdm(os.listdir(TRAIN_DIR)):
    	word_label = img.split('.')[-3]
    	if word_label == 'cat': lable="cats"
    	elif word_label == 'dog': lable="dogs"
    	path = os.path.join(TRAIN_DIR,img)
    	img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    	img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    	cv2.imwrite("./training_data/"+lable+'/'+str(counter)+'.jpg',img)
    	#print("image saved"+str(counter))
    	counter += 1


'''
def process_val_data():
	counter = 1
	for img in tqdm(os.listdir(TEST_DIR)):
		path = os.path.join(TEST_DIR,img)
		img_num = img.split('.')[0]
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
		cv2.imwrite("/media/parth/06C20E27C20E1B95/machine_leaning/examples_pylibreries/keras/val_data/"+str(counter)+'.jpg',img)
		counter += 1

process_val_data()
'''
create_train_test_dir()
