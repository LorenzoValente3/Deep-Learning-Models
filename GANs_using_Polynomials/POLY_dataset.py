import os
import numpy as np
import tensorflow as tf

# +-----------------------+ #
# | Polynomial data class | #
# +-----------------------+ #

class PolynomialData:
    """Polynomial data class: Generation of 2-D images (40x40 for 1-channel) 
       showing polynomial up to a maximum degree in two variables."""
    
    def __init__(self, data_fraction = 1/3, maxdegree = 5 ):
        
        self.size = 40
        self.num_polys = int( 60000 * data_fraction ) 
        self.maxdegree = maxdegree
        self.polydata = None

    def polynomial(self, degree):
        """Evaluate polynomial over grid of size 40x40."""
        coeff = np.random.normal(0,1,(degree+1, degree+1))
        return [[sum([coeff[i,j]*((x/self.size)**i)*((y/self.size)**j)
            for i in range(degree+1) for j in range(degree+1) if (i+j)<=degree]) 
            for x in range(self.size)] for y in range(self.size)]
    
    def train_and_norm(self):
        """Training set of polynomial images of degree <= self.maxdegree and normalize."""
        self.polydata = np.array([self.polynomial(np.random.randint(0,self.maxdegree)) for i in range(self.num_polys)])
        self.polydata = tf.keras.utils.normalize(self.polydata)
    
    def save(self):
        """Save data into an external folder into a .npy file."""
        dir = os.path.join("dataset")
        if not os.path.exists(dir):
            os.mkdir(dir)
        np.save('./dataset/polydata.npy', self.polydata)

if __name__ == '__main__':
    datap = PolynomialData()
datap.train_and_norm()
datap.save()