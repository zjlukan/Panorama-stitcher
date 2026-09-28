import pyiqa

brisque = pyiqa.create_metric('brisque', device='cpu')
score1 = brisque('images/pano1_cropped.png')  # accepts a file path or a tensor
score2 = brisque('images/hugin_pano.png')
print("BRISQUE score 1: " + str(score1))
print("BRISQUE score 2 " + str(score2))