import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from main.ch03.clustering import StemmedCountVectorizer
import pdb

groups = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'comp.windows.x', 'sci.space']

training_data = sklearn.datasets.fetch_20newsgroups(subset="train",categories=groups)

testing__data = sklearn.datasets.fetch_20newsgroups(subset="test", categories=groups) 

vectorizer = StemmedCountVectorizer()

#SHA256:pq9i0piduacfT+i88dANq6LzvQK1JmEV5yCy32MZqfg clement.san@gmail.com

ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC47AtA6+A4SvXE6FwTU5jvuUQdsv/OkZqa//ZpmoXrc3i2NPee7WACOqKpCgU6vYwlE1IPfwzl3DB6+xMuC3AzUmBCY/HjruaYiY7K44NNvasckqnNaqqVmKEMbAsFLi6sSPqij5sRf4hGxJcL//2YE5QBQxitVSe7AIPMcpnCf7KuCSQZ6kQHUU6zhBS/rZOJBzy/1R+yLlvNvgE8GX0qGtq7waGT4Z1R7o4rGzN6lQTJegUb4P5+JlNW4yCXD1m5lQ+WYmdSkpivThxz5pyxhjl+DNd3snzM3xHcqfou7OiYu9emlnIV/7qrVD2FC52yz/gEyl7FkLxJk6ynBL++LiVsJ6dAq1qGOOzZCNPBDqzoyav5SII3dqwmv1C7NVgqtiDuRwPYBd2LZGvz+go+uwr3JP2cIKTRa8az4br6d87Ev8zF/okKFWI7YWvzZM3Sjz6Ycm9EE9fmSnmU91/y18BkqMgY7MbHRnNzZfTOcXycpMl95X8lWLNyinp1IM1qKgKxFZbCAwq6DhLsjBdO7s3+xUHkBpRODz0xMBt0XFYlAXF/D6a6c+vBjJ+xqfqZA4cg9b+XKu7L0jbDMNJ4m1WNBzoh6CY3Jdh877uLUXUm0tM5u57nf4Zc5UgEA3YZAohBjQgyJJUUT2XGax4HFW+gqg97oL9SkqmkV81xuw== clement.san@gmail.com
