import cv2
import numpy as np

class JPEG:
    def __init__(self, qf=50, is_color=False, sample="444") -> None:
        self.quality_factor = qf
        self.is_color = is_color
        self.sample = sample
        self.base_qtable_y = np.array([ [16, 11, 10, 16, 24 , 40 , 51 , 61 ],
                                        [12, 12, 14, 19, 26 , 58 , 60 , 55 ],
                                        [14, 13, 16, 24, 40 , 57 , 69 , 56 ],
                                        [14, 17, 22, 29, 51 , 87 , 80 , 62 ],
                                        [18, 22, 37, 56, 68 , 109, 103, 77 ],
                                        [24, 35, 55, 64, 81 , 104, 113, 92 ],
                                        [49, 64, 78, 87, 103, 121, 120, 101],
                                        [72, 92, 95, 98, 112, 100, 103, 99 ],])
        
        self.base_qtable_c = np.array([ [17, 18, 24, 47, 99 , 99 , 99 , 99 ],
                                        [18, 21, 26, 66, 99 , 99 , 99 , 99 ],
                                        [24, 26, 56, 99, 99 , 99 , 99 , 99 ],
                                        [47, 66, 99, 99, 99 , 99 , 99 , 99 ],
                                        [99, 99, 99, 99, 99 , 99 , 99 , 99 ],
                                        [99, 99, 99, 99, 99 , 99 , 99 , 99 ],
                                        [99, 99, 99, 99, 99 , 99 , 99 , 99 ],
                                        [99, 99, 99, 99, 99 , 99 , 99 , 99 ],])
        self.scaled_qtable_y = None
        self.scaled_qtable_c = None
        self.scale_qtable()
    
    def encode(self, img):
        result = np.zeros((img.shape[0], img.shape[1]))
        size = 8
        for y in range(0, img.shape[0] - size + 1, size):
            for x in range(0, img.shape[1] - size + 1, size):
                result[y:y+size, x:x+size] = self.encode_mcu(img[y:y+size, x:x+size])
        return result
    
    def decode(self, encoded):
        result = np.zeros((encoded.shape[0], encoded.shape[1]))
        size = 8
        for y in range(0, encoded.shape[0] - size + 1, size):
            for x in range(0, encoded.shape[1] - size + 1, size):
                result[y:y+size, x:x+size] = self.decode_mcu(encoded[y:y+size, x:x+size])
        return result
    
    def encode_mcu(self, mcu):
        dct = self.dct(mcu)
        quan = self.quanti(dct)
        return quan
    
    def decode_mcu(self, quan):
        iquan = self.iquanti(quan)
        idct = self.idct(iquan)
        return idct
    
    def dct(self, mcu):
        mcu -= 128
        if self.is_color:
            # return np.stack((np.round(cv2.dct(mcu[:,:,0]-128)), np.round(cv2.dct(mcu[:,:,1]-128)), np.round(cv2.dct(mcu[:,:,2]-128))), axis=2)
            return np.dstack((cv2.dct(mcu[:,:,0]), cv2.dct(mcu[:,:,1]), cv2.dct(mcu[:,:,2])))
        return cv2.dct(mcu)

    def idct(self, iquan):
        if self.is_color:
            return np.dstack((cv2.idct(iquan[:,:,0]), cv2.idct(iquan[:,:,1]), cv2.idct(iquan[:,:,2]))) + 128
        return np.round(cv2.idct(iquan)) + 128
    
    def quanti(self, dct, isCrCb=False):
        if self.is_color:
            return np.dstack((np.round(dct[:,:,0] / self.scaled_qtable_y), np.round(dct[:,:,1] / self.scaled_qtable_c), np.round(dct[:,:,2] / self.scaled_qtable_c)))
        if isCrCb:
            return np.round(dct / self.scaled_qtable_c)
        
        return np.round(dct / self.scaled_qtable_y)
    
    def iquanti(self, quan, isCrCb=False):
        if self.is_color:
            yy = quan[0] * self.scaled_qtable_y
            cr = quan[1] * self.scaled_qtable_c
            cb = quan[2] * self.scaled_qtable_c
            return np.dstack((yy, cr, cb))
        if isCrCb:
            return quan * self.scaled_qtable_c
        
        return quan * self.scaled_qtable_y
    
    def split_16_ycrcb(self, img):
        y0 = img[:8, :8, 0].copy()
        y1 = img[:8, 8:, 0].copy()
        y2 = img[8:, :8, 0].copy()
        y3 = img[8:, 8:, 0].copy()
        cr = img[1::2, ::2, 1].copy()
        cb = img[0::2, ::2, 2].copy()
        return [y0, y1, y2, y3, cr, cb]
    
    def setQF(self, qf):
        self.quality_factor = qf
        self.scale_qtable()
    
    def setColor(self, is_color):
        self.is_color = is_color
        
    def scale_qtable(self):
        scale_factor = 0
        if self.quality_factor < 50:
            scale_factor = 5000 / self.quality_factor
        else:
            scale_factor = 200 - 2 * self.quality_factor
        
        scaled_qtable_Y = np.round((self.base_qtable_y * scale_factor + 50) / 100)
        scaled_qtable_C = np.round((self.base_qtable_c * scale_factor + 50) / 100)
        
        scaled_qtable_Y[np.where(scaled_qtable_Y <= 1)] = 1
        scaled_qtable_C[np.where(scaled_qtable_C <= 1)] = 1
        
        self.scaled_qtable_y = scaled_qtable_Y
        self.scaled_qtable_c = scaled_qtable_C