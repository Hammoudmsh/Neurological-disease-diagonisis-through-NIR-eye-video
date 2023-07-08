import csv
import pandas as pd
import dataframe_image as dfi
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from IPython.display import display



class utilitis:
    def compareTwoList(self, a, b):
        matches = [idx for idx, item in enumerate(zip(a, b)) if item[0] == item[1]]
        matchesNum = len(matches)
        return matches, matchesNum

    def save2csv(self, fileName, data, cols, header = False):
        with open(fileName, 'w+', newline='', encoding='utf-8') as f:
            write = csv.writer(f, delimiter=',')
            if header:
                write.writerow(cols)
            write.writerows(data)
    def find_between(self, s, first, last ):
        try:
            start = s.index( first ) + len( first )
            end = s.index( last, start )
            return s[start:end]
        except ValueError:
            return ""

    def find_between_r(self, s, first, last ):
        try:
            start = s.rindex( first ) + len( first )
            end = s.rindex( last, start )
            return s[start:end]
        except ValueError:
            return ""

    def isContain(self, fileName, fileTypes):
            for ext in fileTypes:
                if  ext not in fileName:
                    return False
            return True

    def show(self, df, nr):
        with pd.option_context('display.max_rows', nr,
                           'display.max_columns', None,
                           'display.width', 800,
                           'display.precision', 3,
                           'display.colheader_justify', 'left'):
            display(df)
        
    def tensor(self, *x):
        tmp = []
        for i in x:
            tmp.append(np.array(i))
    #         tmp.append(tf.convert_to_tensor(i))
        return tmp

    def dataframeAsImage(self, d, path, rowNames, save, colsNames =None):
        df = pd.DataFrame(data=d, index = rowNames, columns = colsNames)
        if save:
            dfi.export(df, path)
        return df
    def showRow(self, display_list, title, size = None):
        # plt.figure()
        fig, ax = plt.subplots(1, len(display_list), figsize = size)
        for i in range(len(display_list)):    
            if display_list[i] is not None:
                ax[i].set_title(title[i])
                ax[i].imshow(tf.keras.utils.array_to_img(display_list[i]));
                ax[i].axis('off')
                plt.close()
        # plt.show()
        return fig
        df = pd.DataFrame(data=d, index = rowNames)
        if save:
            dfi.export(df, path)

    def display(self, display_list, idx = None, num = None, title =  None, size =(10, 10), show = True):    
        if len(display_list[0].shape) in [2,3] :
            f = self.showRow(display_list, title, size = size)
            return f
        else:
            if idx is  None and num is not None or num == 1:
                idx = np.random.randint(0, len(display_list[0]), num)
            fig, ax = plt.subplots(num, len(display_list), figsize = size)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            for j, i in enumerate(idx):
                if j ==0:
                    titles__ = title
                else:
                    titles__ = [""] * len(display_list)
                tmp = []
                for img in display_list:
                    if img is not None and i < len(img):
                        x = img[i]
                    else:
                        x = None
                    tmp.append(x)
                    
                for i in range(len(display_list)):    
                    if tmp[i] is not None:
                        ax[j][i].set_title(titles__[i])
                        if i  in [1,2]:
                            ax[j][i].imshow(tf.keras.utils.array_to_img(tmp[i]), cmap = 'jet');
                        else:
                            ax[j][i].imshow(tf.keras.utils.array_to_img(tmp[i]));
                        ax[j][i].axis('off')
                        ax[j][i].set_aspect('equal')

                        plt.subplots_adjust(wspace=0.1, hspace=0.1)
                        if show == False:
                            plt.close()
            return fig
