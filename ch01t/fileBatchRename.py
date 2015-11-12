import glob, os


class Rename():
	def rename(self):
		dir = 'c:\\test\\'
		pattern = '*.*'
		titlePattern = 'new(%s)'
		print('coucou')
		for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
			title, ext = os.path.splitext(os.path.basename(pathAndFilename))
			os.rename(pathAndFilename, os.path.join(dir, titlePattern % title + ext))


if __name__ == "__main__":
    Rename().rename()

#rename(r'c:\temp\xx', r'*.doc', r'new(%s)')
#The above example will convert all *.doc files in c:\temp\xx dir to new(%s).doc,
# where %s is the previous base name #of the file (without extension).