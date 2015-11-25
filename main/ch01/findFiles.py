import os
import re
import win32api
import shutil
import pdb
import time


extensions = (".jpg", ".jpeg", ".tiff", ".gif", ".png")
#extensions = ".pdf"

rootBackup = os.path.join('E:','\\backupImagenes')

exclude = ("/$Recycle.Bin")


class Files():

    def main(self):
        if os.path.isdir(rootBackup):
            print("Folder {0} already exists".format(rootBackup)) 
           
        else:
            print("Folder {0} does not exist, creating...".format(rootBackup))
            os.makedirs(rootBackup)
        for root, dirs, files in os.walk(r"E:"):
            for file in files:
                print file
        self.findFiles()


    def findFiles(self):
        
        for root, dirs, files in os.walk("/"):
            for file in files:
                if file.endswith(extensions) and exclude not in root:
                    print(os.path.join(root, file))
                    directDirectory = self.getDirectDirectory(root)
                    dstdir =  os.path.join(os.path.join(rootBackup,directDirectory), file)
                    
                    shutil.copy2(os.path.join(root, file), dstdir)
                    #pdb.set_trace()




    def getDirectDirectory(self, dir):
        dirList =  dir.split("\\")
        if len(dirList)>2:
            directDirectory = os.path.join(os.path.join(dirList[-3],dirList[-2]),dirList[-1])
        else if len(dirList)==2 :
            directDirectory = os.path.join(dirList[-2],dirList[-1])
        else :
            directDirectory = dirList[-1] 
        if not os.path.isdir(os.path.join(rootBackup,directDirectory)):
            os.makedirs(os.path.join(rootBackup,directDirectory))
        print directDirectory
        return  directDirectory

       
if __name__=="__main__":
    Files().main()