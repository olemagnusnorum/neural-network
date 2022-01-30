import numpy as np
import random
import matplotlib.pyplot as plt

class Generator:
    """
    A class for generating images of four different figures (crosses, triangles, rectangles, cirlces)
    """

    def generate(self, number=1, size=10, split=(None, None, None), noise=0.0, object_height_range=(10,10), object_width_range=(10,10), centerd=False,  flattend=False):
        """ 
        Main generation function.

        Args:
            number (int): number of nxn pictures to generate
            size (int): size of the nxn images
            split (float, float, float): tuple of the train, valid, test split (train, valid test) if non is given on set is returned
            noise (float): fraction of the noise in the images
            object_height_range (int, int): tuple of the range of the height if the figure in the image
            object_width_range (int, int): tuple of the range of the width if the figure in the image
            centerd (boolean): if the figure is to be centerd in the image
            flattend (boolean): if to represent the image as a vector or a matrix

        Returns:
            tuple of tuples of image sets and lables to the image  
        """
        # np array for the images
        image_batch = np.zeros((number, size, size))
        # labels for the images (0 = cross, 1 = rectangle, 2 = circle, 3 = triangle)
        label_array = np.zeros((number))
        # checking if need to split
        if split[0] != None and split[1] != None and split[2] != None:
            num_train = round(split[0]*number)
            num_valid = round(split[1]*number)
            num_test = round(split[2]*number)
        #canvas for image
        canvas = np.zeros((size, size))
        #iterating throug and "painting" the image
        for i in range(number):
            # seting semi random position for the object by specified width and height
            height = random.randint(*object_height_range)
            width = random.randint(*object_width_range)
            if centerd:
                start_x = size//2 - width//2
                start_y = size//2 - height//2
            else:
                padding_x = size - width
                padding_y = size - height
                start_x = random.randint(0, padding_x)
                start_y = random.randint(0, padding_y)
            r = random.random()
            # choosing object randomly to generate (in big batches this aproches equal number of each object)
            if r <= 0.25:
                image_batch[i] = self._generate_cross(canvas, height, width, start_x, start_y, noise)
                label_array[i] = 0
            elif r <= 0.50:
                image_batch[i] = self._generate_rectangle(canvas, height, width, start_x, start_y, noise)
                label_array[i] = 1
            elif r <= 0.75:
                image_batch[i] = self._generate_circle(canvas, height, width, start_x, start_y, noise)
                label_array[i] = 2
            else:
                image_batch[i] = self._generate_triangle(canvas, height, width, start_x, start_y, noise)
                label_array[i] = 3
            
        # if representation is to be a vector
        if flattend:
            image_batch = image_batch.reshape(number, size*size, 1)
        
        # if there is a specified split
        if split[0] != None and split[1] != None and split[2] != None:
            train_image_set = image_batch[0:num_train]
            temp = label_array[0:num_train]
            train_label_array = np.zeros((4,len(temp)))
            for i in range(len(temp)):
                j = int(temp[i])
                train_label_array[j,i] = 1
            #train_label_array = label_array[0:num_train]
            

            valid_image_set = image_batch[num_train:num_train+num_valid]
            temp = label_array[num_train:num_train+num_valid]
            valid_label_array = np.zeros((4,len(temp)))
            for i in range(len(temp)):
                j = int(temp[i])
                valid_label_array[j,i] = 1
            #valid_label_array = label_array[num_train:num_train+num_valid]

            test_image_set = image_batch[num_train+num_valid:num_train+num_valid+num_test]
            temp = label_array[num_train+num_valid:num_train+num_valid+num_test]
            test_label_array = np.zeros((4, len(temp)))
            for i in range(len(temp)):
                j = int(temp[i])
                test_label_array[j,i] = 1
            #test_label_array = label_array[num_valid:num_train+num_valid+num_test]

            # ((traing_images, label), (valid_images, labels), (test_images, labels) )
            return ((train_image_set, train_label_array), (valid_image_set, valid_label_array), (test_image_set, test_label_array))
            
        return image_batch




    
    def _generate_rectangle(self, canvas, height, width, start_x, start_y, fraction):
        """
        Draws a rectangle and returns it
        """
        image = canvas.copy()
        # making a filled rect
        image[start_y:start_y+height, start_x:start_x+width] = 1
        # erasing the midle
        image[start_y+1:start_y+height-1, start_x+1:start_x+width-1] = 0
        # adding noise
        image = self._add_noise(image, fraction)
        return image

    def _generate_cross(self, canvas, height, width, start_x, start_y, fraction):
        """
        Draws a cross and returns it
        """
        image = canvas.copy()
        center_x = start_x + width//2
        center_y = start_y + height//2
        # making vertical line
        image[start_y:start_y+height, center_x] = 1 
        # making horizontal line
        image[center_y, start_x:start_x+width] = 1
        image = self._add_noise(image, fraction)
        return image
    
    def _generate_circle(self, canvas, height, width, start_x, start_y, fraction):
        """
        Draws a circle and retuns it
        """
        image = canvas.copy()
        # finding radius
        radius = min(width//2, height//2)
        if min(width,height) == 10:
            radius -= 1
        # finding center
        center_x = start_x + width//2
        center_y = start_y + height//2
        #drawing circle
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                # using the equation for a circle to decide where to set 1
                if abs((j-center_x)**2 + (i-center_y)**2 - radius**2) < radius:       
                    image[i,j] = 1
        image = self._add_noise(image, fraction)
        return image

    def _generate_triangle(self, canvas, height, width, start_x, start_y, fraction):

        """
        Draws a triangle and returns it
        """

        # taken from https://www.redblobgames.com/grids/line-drawing.html 
        def lerp(start, end, t):
            return start + t*(end-start)

        def len_diagonal(start, end):
            return max(abs(end[0] - start[0]), abs(end[1] - start[1])) 

        image = canvas.copy()
        # Draw first line
        image[start_y:start_y+height, start_x+width-1] = 1
        # find point
        vertex_x = start_x
        vertex_y = start_y+(height//2)
        # linear interpolate diagonal up
        diagonal_distance = len_diagonal((vertex_x, vertex_y), (start_x+width-1, start_y))
        for i in range(diagonal_distance+1):
            t = 0 if diagonal_distance == 0 else i/diagonal_distance
            lerp_x = round(lerp(vertex_x, start_x+width-1, t))
            lerp_y = round(lerp(vertex_y, start_y, t))
            image[lerp_y, lerp_x] = 1
        
        # linear interpolate diagonal down
        diagonal_distance = len_diagonal((vertex_x, vertex_y), (start_x+width-1, start_y+height-1))
        for i in range(diagonal_distance+1):
            t = 0 if diagonal_distance == 0 else i/diagonal_distance
            lerp_x = round(lerp(vertex_x, start_x+width-1, t))
            lerp_y = round(lerp(vertex_y, start_y+height-1, t))
            image[lerp_y, lerp_x] = 1
        

        image = self._add_noise(image, fraction)
        return image
        
        

    def _add_noise(self, image, fraction):
        """
        Adding noise to an image and returns it
        """
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                # using a random number to decide if the pixel is to be flipped
                if random.random() < fraction:
                    image[i, j] = 1 - image[i, j]
        return image
    


if __name__ == "__main__":
    g = Generator()
    i_set = g.generate(number=100, size=20, split=(0.7, 0.2, 0.1), noise=0.05, object_height_range=(10,20), object_width_range=(10,20), flattend=False)
    print(i_set[0][0][0])
    traning_set = i_set[0][0]
    traning_labels = i_set[0][1]
    #print(traning_set[0])
    #print(traning_labels[0])

    #plt.imshow(i_set[0][0][0])
    #plt.show()


    



    #code to show multiple images
    """
    """
    fig = plt.figure(figsize=(8, 8))
    for i in range(len(traning_set)):
        img = traning_set[i]
        fig.add_subplot(10,10,i+1)
        plt.axis('off')
        plt.imshow(img)
    plt.show()
    
