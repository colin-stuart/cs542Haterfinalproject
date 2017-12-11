import json
from Data import Data

#A class to make saving data a little easier
class CDB:

    def __init__(self):
        #load from filename
        self.data = None


    def save(self, filename):
        #save to file
        f = open(filename,"w")
        j = json.dumps(self.data.get_data())
        print(j,file=f)
        f.close()


    def load(self,filename):
        #load from file
        f = open(filename,"r")
        plain = f.read()
        decrypted = json.loads(plain)
        data = Data()
        data.set_data(decrypted)
        self.data = data
        f.close()

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data.get_data()
