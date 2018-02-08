import os
from model_keras import model

if __name__ == '__main__':
    filenames = [filename.replace('.jpg', '') for filename in os.listdir('inputs/train/')] 
    print(filenames)