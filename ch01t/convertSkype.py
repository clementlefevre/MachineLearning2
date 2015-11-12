import io
import sys
import traceback
import logging
import scipy as sp


logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
contacts = []
global contact
data = []
keyword = ["BEGIN", "VERSION", "N","TEL TYPE=CELL", "TEL TYPE=WORK","TEL;TYPE=HOME" ]


class SkypeConverter():

    def main(self):
        file = self.getFile(sys.argv[1])
        logging.info('File to open : %s', file.name)
        self.getContacts(file)
        self.writeContactFile()
       
    def getFile(self, file):
        if len(file) < 2:
            print("You must enter a .vcf file name")
        else:
            file = io.open(sys.argv[1], encoding="UTF-8") 
            return file

    
    def getContacts(self, file):
        for line in file:
            if len((line.split(":")))==2:
                self.getContact(line)
            line = file.readline()

    def getContact(self,line):
        global contact
        split = line.split(":")
        key = split[0].replace(";"," ")
        value = split[1].replace(";"," ")
        if 'BEGIN' in key:
            contact=[]
            contact.append({'key':key,"value":value
                })
        elif "VERSION" in key :
            contact.append({'key':key,"value":value
                })
        elif key  == "N":
            contact.append({'key':'N',"value":value
                })
            contact.append({'key':"X-SKYPE-DISPLAYNAME","value":value
                })
        elif "TEL TYPE" in key:
            value = self.getTel(value)
            contact.append({'key':"X-SKYPE-PSTNNUMBER","value":value
                })
        elif  "EMAIL" in key:
            contact.append({'key':"EMAIL","value":value
                            })
        elif  "END" in key :
            contact.append({'key':"REV","value":"01151106T112822Z\n"
                            })
            contact.append({'key':key,"value":value
                })
            contact[3], contact[4] = contact[4],contact[3]
            contacts.append(contact)

            


    def getTel(self, value):
        try:

            value = '+{0}'.format(value.strip(' '))
            value = value.replace('-', '')
            value = value.replace(' ', '')
            value = value.replace('(', '')
            value = value.replace(')', '')
            value = value.replace(u'\xa0', '')


        except Exception :
            traceback.print_exc()
            logging.error(value)
        return value

    def writeContactFile(self):
        with io.open('output.vcf', 'w',encoding="utf8") as file:
            for contact in contacts:
                for arg in contact:
                    try:
                        textToWrite = arg['key'] +unicode(":")+unicode(arg["value"])
                        file.write(textToWrite)
                    except Exception :
                        traceback.print_exc()
                file.write(u"\n")

if __name__ == "__main__":
    SkypeConverter().main()