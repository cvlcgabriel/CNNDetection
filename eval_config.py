from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
dataroot = '/content/CNNDetection/dataset/test'

# list of synthesis algorithms
vals = ['stylegan', 'deepfake']

# indicates if corresponding testset has multiple classes
multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

# model
model_path = '/content/CNNDetection/weights/blur_jpg_prob0.5.pth'
