
#!/usr/bin/python
"""
create dataset of 1024 elements
with the proper class
"""
import numpy as np
import os
#create a binary CSV file for Boolean function
result=open("datayoun.csv", 'w')
zero=0
un=1
with result as outfile:
    for i1 in range(2):
        for i2 in range(2):
            for i3 in range(2):
                for i4 in range(2):
                    for i5 in range(2):
                        for i6 in range(2):
                            for i7 in range(2):
                                for i8 in range(2):
                                    for i9 in range(2):
                                        for i10 in range(2):
                                            if (( (i1 and i2) or (i4 and i6) ) == 1):
                                                outfile.write(str(i1)+","+str(i2)+","+str(i3)+","+str(i4)+","+str(i5)+","+str(i6)+","+                                            str(i7)+","+str(i8)+","+str(i9)+","+str(i10)+",1 \n")
                                            else:
                                                outfile.write(str(i1)+","+str(i2)+","+str(i3)+","+str(i4)+","+str(i5)+","+str(i6)+","+ str(i7)+","+str(i8)+","+str(i9)+","+str(i10)+",0 \n")
                                            
