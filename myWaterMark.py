import numpy as np
import matplotlib.pyplot as plt
import cv2


def bgr2rgb(f):
    b, g, r = cv2.split(f)
    return cv2.merge([r,g,b])


def ti(f):
    return f.astype('int')

img = cv2.imread('output.JPG')
b, g, r = cv2.split(img)
_fft = np.fft.fft2(b)
temp = _fft

def processor(_fft):
    for i in range(97, 108):
        for j in range(62, 143):
            _fft[i][j] = 50000

    for i in range(302, 313):
        for j in range(267, 348):
            _fft[i][j] = 50000

processor(_fft)
_reverse = np.fft.ifft2(_fft)

_reverse = _reverse.astype(np.int32)
r = r.astype(np.int32)
g = g.astype(np.int32)

output = cv2.merge([_reverse, g, r])
cv2.imwrite('output.JPG', output)
cv2.namedWindow('test', cv2.WINDOW_AUTOSIZE)
cv2.imshow('test', ti(_fft))
cv2.waitKey(0)
cv2.destroyAllWindows()


#plt.subplot(1,2,1)
#plt.imshow(bgr2rgb(img))
#plt.subplot(1,2,2)
#plt.imshow(output)
#plt.show()
