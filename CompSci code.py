import json
from os import replace
import numpy as np
from numpy.lib.function_base import average, delete
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest
import random
import math
from pytextdist.vector_similarity import cosine_similarity
from pytextdist.edit_distance import levenshtein_distance, levenshtein_similarity
from scipy.spatial.kdtree import distance_matrix
from similarity.normalized_levenshtein import NormalizedLevenshtein
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import resample
import matplotlib.pyplot as plt

#Load Data
dataframe = open('TVs-all-merged.json',)
data = json.load(dataframe)

#Calling this method will run both LSH and MSM 
def ProcedureLSHandMSM(data, N, nrows, listofsampleproducts): 
  
  #STEP 0: saving information for evaluation purposes and cleaning data

  #getting a list of the Model IDs of the current sample
  def getlistofModelIDs():
    listofModelIDs =[]
    listofsampleModelID = []
    for h in data:
      for i in data[h]:
        listofModelIDs.append(h)
    for i in listofsampleproducts:
      listofsampleModelID.append(listofModelIDs[i])
    return listofsampleModelID
  
  listofsampleModelID = getlistofModelIDs()  

  #print("length of the model IDs: ", len(listofsampleModelID))

  #Getting a list with the product data for every product in the sample
  def getlistwithproductdata():
    listwithproductdata = []
    listwithsampleproductdata = []
    for h in data:
      for i in data[h]:
        listwithproductdata.append(i)
    for i in listofsampleproducts:
      listwithsampleproductdata.append(listwithproductdata[i])
    return listwithsampleproductdata
  listwithsampleproductdata = getlistwithproductdata()

  #Perform data cleaning on the dataset. Measurements of the same unit should be represented in an equal manner (inch, hz, lb). 
  #We also remove spaces between numeric tokens and units, and set all characters to lower case letters. 
  def cleaningData(data):
    for h in data:
     for i in data[h]:
      for key, value in i.items():
        if key == "title":
          i[key] = value.lower()
          i[key] = i[key].replace("\"", " inch") #werkt niet?
          i[key] = i[key].replace("inches", "inch")
          i[key] = i[key].replace("Inch", "inch")
          i[key] = i[key].replace("-inch", "inch")
          i[key] = i[key].replace(" Hz", "Hz")
          i[key] = i[key].replace("lb.", "lb")
          i[key] = i[key].replace("lbs.", "lb")
          i[key] = i[key].replace("pounds", "lb")
          i[key] = i[key].replace(" pounds", "lb")
          i[key] = i[key].replace(" lbs", "lb")
          i[key] = i[key].replace("-Inch", "inch")
          i[key] = i[key].replace(" lb", "lb")
          i[key] = i[key].replace(" inch", "inch")  

    for h in data:
     for i in data[h]:
      for key, value in i.items():
          if key == "featuresMap":
            for key2, value2 in value.items():
                value[key2] = value[key2].lower()
                value[key2] = value[key2].lower()
                value[key2] = value[key2].replace("Inch", "inch")
                value[key2] = value[key2].replace("inches", "inch")
                value[key2] = value[key2].replace("\"", "inch") #werkt niet?
                value[key2] = value[key2].replace("-inch", " inch")
                value[key2] = value[key2].replace("-Inch", "inch")
                value[key2] = value[key2].replace(" inch", "inch")
                value[key2] = value[key2].replace(" Hz", "Hz")
                value[key2] = value[key2].replace("lbs.", "lb")
                value[key2] = value[key2].replace("lb.", "lb")
                value[key2] = value[key2].replace("pounds", "lb")
                value[key2] = value[key2].replace(" pounds", "lb")
                value[key2] = value[key2].replace("lbs", "lb")
                value[key2] = value[key2].replace(" lb", "lb")

    return data
  
  data = cleaningData(data)

  #STEP 1: Obtaining Binary Vectors 
  modelWords = [] #to store all title model words

  #adding all Model Words from the title to the list of model words. 
  def addMWFromTitle(data):
    for h in data:
     for i in data[h]:
      for key, value in i.items():
          if key == "title":
            # value = value.lower()
            # value = value.replace("\"", " inch") #werkt niet?
            # value = value.replace("inches", "inch")
            # value = value.replace("Inch", "inch")
            # value = value.replace("-inch", "inch")
            # value = value.replace(" Hz", "Hz")
            # value = value.replace("lb.", "lb")
            # value = value.replace("lbs.", "lb")
            # value = value.replace("pounds", "lb")
            # value = value.replace(" pounds", "lb")
            # value = value.replace(" lbs", "lb")
            # value = value.replace("-Inch", "inch")
            # value = value.replace(" lb", "lb")
            # value = value.replace(" inch", "inch")
            user_string = value
            pattern = '([a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*)'
            result = re.findall(pattern, user_string)
            for word in result:
              if word[0] not in modelWords:
                modelWords.append(word[0])
  addMWFromTitle(data) 

  #adding all relevant values from the features: Maximum Resolution, Dynamic Contrast Ratio, Speaker Output Ratio and TV Type. These features contain
  # .............................
  def addMWFromRelevantFeatures(data):
    for h in data:
     for i in data[h]:
      for key, value in i.items():
          if key == "featuresMap":
            for key2, value2 in value.items():
              if key2 == "Maximum Resolution":
                # value2 = value2.lower()
                # value2 = value2.replace("Inch", "inch")
                # value2 = value2.replace("inches", "inch")
                # value2 = value2.replace("\"", "inch") #werkt niet?
                # value2 = value2.replace("-inch", " inch")
                # value2 = value2.replace("-Inch", "inch")
                # value2 = value2.replace(" inch", "inch")
                # value2 = value2.replace(" Hz", "Hz")
                # value2 = value2.replace("lbs.", "lb")
                # value2 = value2.replace("lb.", "lb")
                # value2 = value2.replace("pounds", "lb")
                # value2 = value2.replace(" pounds", "lb")
                # value2 = value2.replace("lbs", "lb")
                # value2 = value2.replace(" lb", "lb")
                if value2 not in modelWords:
                    modelWords.append(value2)
              if key2 == "Dynamic Contrast Ratio":
                value2 = value2
                if value2 not in modelWords:
                    modelWords.append(value2)
              if key2 == "Speaker Output Power":
                value2 = value2
                if value2 not in modelWords:
                    modelWords.append(value2)
              if key2 == "TV Type":
                value2 = value2.split()[0]
                if value2 not in modelWords:
                    modelWords.append(value2)
              
  addMWFromRelevantFeatures(data)
  print("amount of model words:", len(modelWords))

  #create a binary vector for every product to fill with 0's and 1's. Every array now has a name corresponding to the index of the product
  def obtainBinaryVectors(modelWords):
    arrays = {}
    arraysSample = {}
    index =0
    for h in data:
      for i in data[h]:
        arrays[index] = [0] * len(modelWords)
        index = index+1
    index =0
    for h in data:
     for i in data[h]:
      for word in modelWords:
        # title = i["title"].lower()
        # title = title.replace("Inch", "inch")
        # title = title.replace("inches", "inch")
        # title= title.replace("\"", "inch") #werkt niet?
        # title = title.replace("-inch", " inch")
        # title = title.replace("-Inch", "inch")
        # title = title.replace(" inch", "inch")
        # title = title.replace(" Hz", "Hz")
        # title = title.replace("lbs.", "lb")
        # title = title.replace("lb.", "lb")
        # title = title.replace("pounds", "lb")
        # title = title.replace(" pounds", "lb")
        # title = title.replace("lbs", "lb")
        # title = title.replace(" lb", "lb")
        if word in i["title"]:
          arrays[index][modelWords.index(word)] = 1
      index = index+1
    index=0
    for h in data:
        for i in data[h]:
          for key, value in i.items():
            if key == "featuresMap":
              for key2, value2 in value.items():
               for word in modelWords:
                  # value2 = value2.lower()
                  # value2 = value2.replace("Inch", "inch")
                  # value2 = value2.replace("inches", "inch")
                  # value2= value2.replace("\"", "inch") #werkt niet?
                  # value2 = value2.replace("-inch", " inch")
                  # value2 = value2.replace("-Inch", "inch")
                  # value2 = value2.replace(" inch", "inch")
                  # value2 = value2.replace(" Hz", "Hz")
                  # value2 = value2.replace("lbs.", "lb")
                  # value2 = value2.replace("lb.", "lb")
                  # value2 = value2.replace("pounds", "lb")
                  # value2 = value2.replace(" pounds", "lb")
                  # value2 = value2.replace("lbs", "lb")
                  # value2 = value2.replace(" lb", "lb")
                  if word in value2:
                    arrays[index][modelWords.index(word)] = 1
          index = index+1
    sampleindex = 0
    for i in listofsampleproducts:
      arraysSample[sampleindex] = arrays[i] 
      sampleindex = sampleindex +1

    return arraysSample
    
  #print("should be 1624 again:", len(obtainBinaryVectors(modelWords)))
  binaryvectors = obtainBinaryVectors(modelWords)
  listofsampleproductsindeces = list(binaryvectors.keys()) #store the indeces of the products in the sample to find corresponding Model ID's later for evaluation
  #print("this should be 1624 to work:", len(listofsampleproductsindeces))

  #STEP 2: Minhashing
  n = N #number of minhashes will be approximately half of the number of original rows of the binary vectors. 
  M = len(binaryvectors) #Declaring columns of signature matrx as number of products.

  # p must be slightly bigger than the maximum of x (prime) (x is r, dus hoeveel rows/products we gaan doen)
  p = 1009 #2013 model words / 2 = 1006. Rond af naar 1000
  def hash_function(a,b,p,x):
    result = (a+b*x)%p
    return result #nog seed nodig?

  #Performs the minhashing, generating a signature matrix with the signatures for each product. We first generate n permutations of the rows, and then look for the first row with a one. 
  #The number of the first row with a one will be placed in the signature matrix. 

  def Minhash(n,M):
    Smatrix = np.full((n,M),10000)
    permutations = {}
    normal_order = []
    for numbers in range(len(modelWords)): 
      normal_order.append(numbers)
    for permutation in range(n): #create 1000 permutations of row orders
      permutations[permutation] = np.random.permutation(normal_order) 
    for product in range(len(listofsampleproductsindeces)): #for every product shuffle the rows under certain permutation 
      for permutation in range(n):
        for wordnumber in permutations[permutation]:
          if binaryvectors[product][wordnumber] ==1:
            Smatrix[permutation][product] = wordnumber
    return Smatrix

  Smatrix = Minhash(n,M)

  print(Smatrix) #print and do Smatrix Hash
  print(len(Smatrix)) #1000 rows rows
  print(len(Smatrix[2])) #1624 products/columns


  #STEP 3: LSH
  n = n #the amount of minhashes
  rows = nrows #the amount of rows per band
  bands = int(n/rows) #number of bands 
  t = (1/bands)**(1/rows) #t-value

  #function to pslit a vector (column of signature matrix) into b bands, consisting of r rows
  def divide_vector(signature, b):
    r = int(len(signature)/b)
    sub_vectors = []
    for i in range(0,len(signature), r):
      sub_vectors.append(signature[i : i+r])
    return sub_vectors

  #function to add the row numbers of a certain band together to form a string
  def hashtostring(band):
    string = ""
    for int in band:
      string += str(int)
    return string

  #function that creates the duplicate matrix. First, it divides the signature matrix into bands of r rows. Then, it compares the bands of products in the same row. If the bands of two products are at least 
  #equal in one of the rows, the products will be considered as candidate duplicates.
  def createDuplicateMatrix(): 
    buckets = {}
    subvectors = {}
    for p in range(len(listofsampleproductsindeces)):
      buckets[p] = [0] * bands
    for p in range(len(listofsampleproductsindeces)): #voor elk product sla je de subvectors/bands op 
      subvectors[p] = divide_vector(Smatrix[:,p], bands) #elk product heeft nu een array gevuld met arrays van bands 
    def putBucketsInDictionary(product):
      for band in range(bands):
        buckets[product][band] = hashtostring(subvectors[product][band])
    for product in range(len(listofsampleproductsindeces)):
      putBucketsInDictionary(product)
    print("should be 1624", len(buckets)) #dit zijn hoeveel producten er zijn 
    print("amount of bands:", len(buckets[2])) 
    duplicateMatrix = np.zeros((len(listofsampleproductsindeces),len(listofsampleproductsindeces)))
    for product1 in range(len(listofsampleproductsindeces)):
      for product2 in range(len(listofsampleproductsindeces)):
        for i in range(bands):
          if buckets[product1][i]== buckets[product2][i]:
            duplicateMatrix[product1][product2] =1
    return duplicateMatrix
  duplicateMatrix = createDuplicateMatrix()
  duplicateMatrixTriangular = np.triu(duplicateMatrix,1)

  #function that counts how many candidate duplicate pairs we have identified.
  def countCandidatePairs():
    numberofpossibleduplicates = 0
    #to see amount of duplicates
    for i in range(len(listofsampleproductsindeces)):
      for j in range(len(listofsampleproductsindeces)):
        if duplicateMatrixTriangular[i][j] ==1:
          numberofpossibleduplicates = numberofpossibleduplicates+1
    return numberofpossibleduplicates
  print("t:", t)
  print("maximum number of comparisons:", (((len(listofsampleproductsindeces)*len(listofsampleproductsindeces))-len(listofsampleproductsindeces))/2))
  print("number of comparisons:", countCandidatePairs()) 

  #LSH evaluation Returns the amount of actual duplicates found, which is checked by comparing the model IDs of candidate duplicate pairs. 
  def LSHEvaluation():
    duplicatesfound = 0
    for i in range(len(listofsampleproductsindeces)):
      for j in range(len(listofsampleproductsindeces)):
        if duplicateMatrixTriangular[i][j] ==1:
          if listofsampleModelID[i] == listofsampleModelID[j]:
            duplicatesfound = duplicatesfound+1
    return duplicatesfound
  
  #Returns the amount of duplicates that were to be found, which we can check based on the amount of products in the sample with the same Model ID. 
  duplicatesToBeFound = 0
  for i in range(len(listofsampleproductsindeces)):
    for j in range(len(listofsampleproductsindeces)):
      if listofsampleModelID[i] == listofsampleModelID[j]:
        duplicatesToBeFound = duplicatesToBeFound+1
  duplicatesToBeFound = (duplicatesToBeFound - len(listofsampleproductsindeces))/2

  print("zoveel echte duplicate pairs heb ik gevonden:", LSHEvaluation())
  print("fraction of comparisons:", (countCandidatePairs() / (((len(listofsampleproductsindeces)*len(listofsampleproductsindeces))-len(listofsampleproductsindeces))/2)))
  print("pair completeness:", (LSHEvaluation() / 399) )
  print("pair quality:", (LSHEvaluation() / countCandidatePairs() ))

#STEP 4: MSM only on products assigned as duplicates

  most_common_brands = ["Acer","Admiral","Admiral Overseas Corporation","Advent","Adyson","Asianet Digital LED TV","Agath","Agrexsione","Aiwa","Akai","Akari","Akurra","Alba","Amplivision","Amstrad","Andrea Electronics","Anitech","Apex Digital","Arcam","Arena","Argosy Radiovision","Arise India","AGA","Astor","Asuka","Atlantic","Atwater Television","Audar","Automatic Radio Manufacturing","Audiovox","AVEL","AVol","AWA","Bace Television","Baird","Bang & Olufsen","Baumann Meyer","Beko","BenQ","Bell Television","BelTek Tv","Bharat","Beon","Binatone","BiSA","Bitova electronika","Blaupunkt","BLUE Edge","Blue Sky","Blue Star","Bondstec","BOSE","BPL india lmt","Brandt","Brionvega","Britannia","BrokSonic","BSR","BTC","Bush","Calbest Electronics","Caixun","Capehart Electronics","Carrefour","Cascade","Cathay","Cello Electronics","Centurion","Certified Radio Labs","Cenfonix","CGE","Changhong","ChiMei","Cimline","Citizen","Clairetone Electric Corporation","Clarivox","Clatronic","CloudWalker","Coby","Colonial","Color Electronics Corporation","Compal Electronics", "Conar Instruments","Condor","Conrac","Conrac","Contec","Continental Edison","Cortron Industries","Cossor","Craig","Crown","Crystal","CS Electronics","CTC","Curtis Mathes Corporation","Cybertron","Daewoo","Dainichi","Damro","Dansai","Dayton","De Graaf","Decca","Deccacolour","Defiant","Dell","Delmonico International Corporation","Diamond Vision","Diboss","Digihome","Dixi", "Dual","Dual Tec","Dumont","DuMont Laboratories","Durabrand","Dyanora","Dynatron","Dynex","Edler","Electron","Electronics Corp.","English Electric","EKCO","Elbe","Electrohome","Element","Elin","Elite","Elta","Emerson","Emerson","EMI","Erres","Expert","Farnsworth","Ferguson Electronics","Ferranti","Fidelity Radio","Finlandia","Finlux","FIRST","Firstline","FisherElectronics","Fleetwood","Flint","Formenti","Frontech","Fujitsu","Funai","GC","Geloso general Electric" ,"General Electric Company","General Gold","Geloso","Genexxa","GoldStar","Goodmans Industries","Gorenje","GPM","Gradiente","Graetz","Granada","Grandin","Grundig","Haier","Hallicrafters","Hannspree","Hanseatic","Hantarex","Harvard International","Harvey Industries","Haver Electric","HCM","Healing","Helkama","Helvar","Heath Company/Heathkit,Hesstar","Hinari Domestic Applicanes","Hisawa","HMV","Hisense","Hitachi","HKC","Hoffman Television","Horizont","Howard Radio","Huanyu","Hypson","Ice","Ices","Inelec","ITS","ITT Corporation","ITT-KB","ITT-SEL","Imperial","INB","Indiana","Ingelen","Inno Hit","Innovex","Insignia","Interfunk","Intervision","Isukai","IZUMI","Jensen Loudspeakers","JMB","Joyeux","Kaisui","Kamacrown","Kane Electronics Corporation","Kapsch","Kathrein","Kendo","Kenmore","Kent Television","Khind","Kingsley","KIVI","Kloss Video","Kneissel","Kogan","Kolster-Brandes","Konka","Korpel","Koyoda","Kreisler","KTC","Lanix","Le.com","Leyco","LG","Liesenkötter" ,"Linsar","LLoyd" ,"Loewe", "Luma" ,"Luxor" ,"M Electronic","MTC" ,"Magnadyne" ,"Magnafon" ,"Magnasonic","Magnavox","Magnavox","Maneth","Marantz","Marconiphone","Mark","Matsui","Mattison Electronics","McMichael Radio","Mediator","Memorex","Micromax","Mercury-Pacific","Metz","Minerva","Minoka","Mirc Electronics","Mitsubishi","Mivar","Mi TV","Mobile360 Tv","Motorola","Multitech","Muntz","MT Logic","Murphy Radio","NASCO ELECTRONICS","NEC","Neckermann","Nelco","NEI","NEOS","NetTV","Nikkai","Nobliko","Nokia","Nordmende","North American Audio","Olympic Radio and Television","Oceanic","oCosmo","OK tv","Olevia","One","OnePlus","Onida","Onwa","Orion","Orion","Osaki","Oso","Osume","Otake","Otto Versand","Palladium","Panama", "Panasonic","Pathe Cinema","Pathe Marconi","Pausa","Perdio","Pensonic","Peto Scott","Philco","Philips","Philmore Manufacturing","Phonola","Pilot Radio","Pilot Radio Corporation","Pioneer","Planar Systems","Polar","Polaroid","Profilo Holding","Profex","Prima","Privé","ProLine","ProScan","ProTech","Pulser","Pye","Pyle","PyxScape","Quasar","Quelle","Questa","R-Line", "REI","Radiola","Radiola","RadioMarelli","RadioShack","Rank Arena","Ravenswood","Rauland Borg","RBM","RCA","RCA","RFT","RGD","Roadstar","Rolls","Rolsen Electronics","Rubin","SABA","Saccs","Saisho","Salora", "Sambers","Samsung","Sanabria","Sandra","Sansui","Sanyo","SBR","Sceptre","Schaub Lorenz","Schneider","Sears","SEG","SEI","Sei-Sinudyne","Seiki Digital","Selco","Sèleco","Sentra","Setchell Carlson","Seura","Shinco","Shorai","Siarem","Siemens","Silo Digital","Skywalker","Silvertone","Sinudyne","Skyworth","Sobell","Solavox","Sonitron","Sonodyne","Sonoko","Sonolor","Sonora","Sontec","Sony","Soyo","Soundwave","Softlogic","Sparc","Stern-Radio Staßfurt","Stromberg Carlson","Stewart-Warner","SunLite TV","Sunkai","Susumu","Supersonic","Supra","Sylvania","Symphonic Electronic Corp","Symphonic Radio and Electronics","Sysline","Tandy","Technika TV","Tatung Company","TCL","Tec","Tech-Master","Technema","Technics","Technisat","Tecnimagen","Technika","TECO","Teleavia","Telebalt","Telefunken","Telemeister","Telequip","Teletech","Teleton","Teletronics","Television Inc.","Temp","Tensai","Texet","Thomson SA","Thorn Electrical Industries","Thorn EMI","TMA","Tomashi","Toshiba","TPV Technology","TP Vision","Transvision","Trav-Ler Radop","Travelers Electronics Co.","Trinium Electronics Philippines","Triumph","Uher","Ultra","Ultravox","Unitech","United States Television Manufacturing Corp.","Universum","Upstar","Vestel","Videocon","Videosat","Videotechnic","Videoton","Viewsonic","Vision","Vistron","Vizio","Veon","Vu","V WORLD","Warwick","Walton","Watson","Watt Radio","Wells-Gardner","Westinghouse","Westinghouse Digital","Weston Electronics","White-Westinghouse","Witjas","Wybor","X2GEN","Xiaomi","Yamazen","Yara","YC","Yoko","Zanussi","Zenith Radio","Zonda"]
  for i in range(len(most_common_brands)):
    most_common_brands[i] = most_common_brands[i].lower()
  
  #returns true if two products are of a different brand
  def diffBrand(pi, pj): 
    brand1 = ""
    brand2 = ""
    for key1, value1 in listwithsampleproductdata[pi].items():
      if key1 == "title":
        title1 = value1
        for word in most_common_brands:
          if word in title1:
            brand1 = word
    for key2, value2 in listwithsampleproductdata[pj].items():
      if key2 == "title":
        title2 = value2
        for word in most_common_brands:
          if word in title2:
            brand2 = word
    if brand1 != brand2:
      return True
    else:
      return False

  #calculates the number of different q-gram occurences in two arrays of words, which is the qgramdistance.
  def qGramDistance(s1,s2): 
    different =0
    for i in range(len(s1)):
        if s1[i] not in s2:
          different = different+1
    for j in range(len(s2)):
        if s2[j] not in s1:
          different = different+1
    return(different)

  #calculates the q-gram similarity of two strings
  def calcSim(q,r): 
    char = 3
    qtokens = [q[i:i+char] for i in range(len(q)-2)]
    rtokens = [r[j:j+char] for j in range(len(r)-2)]
    if len(qtokens) ==0: #in case a word is less than 3 characters
      qtokens = [q]
    if len(rtokens) ==0: #in case a word is less than 3 characters
      rtokens=[r]
    n1 = len(qtokens) #amount of tokens 
    n2 = len(rtokens)
    distance = qGramDistance(qtokens,rtokens)
    return (n1+n2-distance) / (n1+n2)

  #calculates the percentage of matching model words from two sets of model words
  def mw(C,D): 
    matchingwords = 0
    if len(C) and len(D) > 0:
      for word1 in C:
        for word2 in D:
          if word1 == word2:
            matchingwords = matchingwords+1
      percentage = matchingwords / len(C)
      return percentage
    else:
      return 0

  #
  def getAllKeyValues(i):
    pairs = {}
    for key1, value1 in listwithsampleproductdata[i].items():
              if key1 != "featuresMap":
                newvalue = value1.replace(" ", "")
                newvalue = newvalue.replace(",", "")
                newvalue = newvalue.replace("/", "")
                newvalue = newvalue.replace("\"", "")
                newvalue = newvalue.lower()
                pairs[key1] = newvalue
    for key2, value2 in listwithsampleproductdata[i].items():
              if key2 == "featuresMap":
                for key3, value3 in value2.items():
                  newvalue = value3.replace(" ", "")
                  newvalue = newvalue.replace(",", "")
                  newvalue = newvalue.replace("/", "")
                  newvalue = newvalue.replace("\"", "")
                  newvalue = newvalue.lower()
                  pairs[key3] = newvalue
    return pairs

  #calculates the minimum number of product features that product i and j contains
  def minFeatures(pi, pj):
    minfeatures=0
    featurespi = 0
    featurespj=0
    for key1,value1 in getAllKeyValues(pi).items():
      featurespi = featurespi+1
    for key2, value2 in getAllKeyValues(pj).items():
      featurespj = featurespj+1
    minfeatures = min(featurespi, featurespj)
    return minfeatures

  #retrieves all models words from the values of attributes of a product
  def exMW(p): 
    modelwords = []
    #substrings = [".", ",", "\"", "-", ":", "/", "Hz", "lbs.", "mm", "+", "'", "(", ")", "0W", "0p", "0B", ";", "mW"]
    for key, value in p.items():
      for string in value.split():
        if not string.isalpha() and not string.isnumeric():
              modelwords.append(string)
    return modelwords

  #returns the levenhstein distance between two strings
  normalized_levenshtein = NormalizedLevenshtein()
  def avgLvSim(X,Y):
    normalized_levenshtein = NormalizedLevenshtein()
    return normalized_levenshtein.distance(X,Y)

  #Returns the average Levenhstein similarity between two sets of model words
  def avgLvSimMW(A,B): #input is twee sets van model words. Voeg alleen degene die op elkaar lijken samen. Steeds pairwise vergelijken
    normalized_distances_averages =0
    total_weight = 0
    for a in A:
        for b in B:
          texta = ''.join([i for i in a if not i.isdigit()])
          textb = ''.join([j for j in b if not j.isdigit()])
          numbera= ''.join([k for k in a if k.isdigit()])
          numberb = ''.join([l for l in b if l.isdigit()]) 
          if normalized_levenshtein.distance(texta,textb)  < 0.5 and numbera ==numberb:
            normalized_distances_averages = normalized_distances_averages + (1-normalized_levenshtein.distance(a,b)) * (len(a)+len(b))
            total_weight = total_weight + len(a) + len(b)
    if total_weight > 0:
      modelWordSimVal = normalized_distances_averages / total_weight
      return modelWordSimVal
    else:
      return 0
      
  #implements the TMWM algorithm
  def TMWMSim(pi,pj,alpha,beta, delta, epsilon):  
      normalized_levenshtein = NormalizedLevenshtein()
      for key1, value1 in listwithsampleproductdata[pi].items():
              if key1 == "title":
                title1 = value1
      for key2, value2 in listwithsampleproductdata[pj].items():
              if key2 == "title":
                title2 = value2
      nameCosineSim = cosine_similarity(title1, title2)
      if nameCosineSim > alpha:
        return 1
      modelWordsA = exMW(getAllKeyValues(pi))
      modelWordsB = exMW(getAllKeyValues(pj))
      for a in modelWordsA:
        for b in modelWordsB:
          a = ''.join([i for i in a if not i.isdigit()])
          b = ''.join([j for j in b if not j.isdigit()])
          if normalized_levenshtein.distance(a,b)  < 0.5:
            for c in modelWordsA:
              for d in modelWordsB:
                c = ''.join([k for k in c if k.isdigit()])
                d = ''.join([l for l in d if l.isdigit()]) 
                if c !=d:
                  return -1
      finalNameSim = beta * nameCosineSim + (1-beta) * (1-avgLvSim(title1,title2))
      modelWordSimVal = avgLvSimMW(modelWordsA, modelWordsB)
      if modelWordSimVal > 0:
        finalNameSim = delta * modelWordSimVal + (1-delta) * finalNameSim

      return finalNameSim 


  #performs the gglomerative clustering, which merge until smallest dissimilarity between clusters is higher than epsilon
  def hClustering(dist, epsilon):
    cluster = AgglomerativeClustering(n_clusters = None, affinity = "precomputed", linkage = "complete", distance_threshold = epsilon)
    cluster.fit(dist)
    result = cluster.fit_predict(dist)

    df_dist = pd.DataFrame(data=result,columns=['Cluster'])
    df_dist['modelID']= listofsampleModelID

    grouped_df = df_dist.groupby("Cluster")
    cluster = grouped_df['modelID'].apply(list)
    cluster = cluster.reset_index()
    amountofclusters = len(np.unique(result))

    return cluster
    
  #performs the MSM algorithm, returning the found clusters of duplicates
  def MSM(alpha, beta, gamma, epsilon, mu, delta):
    distanceMatrix = np.zeros((len(listofsampleproductsindeces),len(listofsampleproductsindeces))) 
    for i in range(len(listofsampleproductsindeces)):
      for j in range(len(listofsampleproductsindeces)):
        if duplicateMatrixTriangular[i][j] ==0:
          distanceMatrix[i][j] = 1000000000
        elif duplicateMatrixTriangular[i][j] ==1:
             if diffBrand(i,j):
              distanceMatrix[i][j] = 1000000000
        else:   
             sim = 0
             avgSim = 0
             m=0
             w=0
             nmki = getAllKeyValues(i)
             nmkj = getAllKeyValues(j)
             for keyq, valueq in list(nmki.items()):
              for keyr, valuer in list(nmkj.items()):
                  keySim = calcSim(keyq, keyr)
                  if keySim > gamma:
                    valueSim = calcSim(valueq, valuer)
                    weight = keySim
                    sim = sim+weight*valueSim
                    m = m+1
                    w = w+weight
                    if keyq in nmki:
                      del nmki[keyq]
                    if keyr in nmkj:
                      del nmkj[keyr]
             if w>0:
               avgSim = sim/w
             mwPerc = mw(exMW(nmki), exMW(nmkj))
             titleSim= TMWMSim(i,j,alpha,beta, delta, epsilon) 
             hSim = 0
             if titleSim == -1:
                theta1 = m/minFeatures(i,j)
                theta2 = 1-theta1
                hSim = theta1 * avgSim + theta2 * mwPerc
             else:
                theta1 = (1-mu) * m/minFeatures(i,j)
                theta2 = 1-mu-theta1
                hSim = theta1 * avgSim + theta2 * mwPerc + mu * titleSim   
             distanceMatrix[i][j] = 1-hSim
    sum = 0
    distanceMatrixTriangular = np.triu(distanceMatrix,1) 
    for i in range(len(distanceMatrixTriangular)):
      for j in range(len(distanceMatrixTriangular)):
          if j<=i:
            distanceMatrixTriangular[i][j] = 1000000000 #set all distances under the diagonal to infinity
    for i in range(len(listofsampleproductsindeces)):
      for j in range(len(listofsampleproductsindeces)):
        if distanceMatrixTriangular[i][j] < 1 and distanceMatrixTriangular[i][j] > 0:
          sum = sum+1
    print("amount of normal distances:", sum)
    print("distance matrix:", distanceMatrixTriangular)
    return hClustering(distanceMatrixTriangular, epsilon) 

  clusterResults = MSM(0.6, 0, 0.75, 0.5, 0.65,0.4)
  print("cluster results:", clusterResults)
  #for i in range(1000):
    #if clusterResults[i] > 0:
    # print(clusterResults[i])

  duplicatesFound = LSHEvaluation() 
  numberOfComparisons = countCandidatePairs()



  return [duplicatesFound, numberOfComparisons, duplicatesToBeFound, clusterResults]

#STEP 5: EVALUATION
#bootstrap sample size: same as orginal dataset!!
#5 bootstrap samples
numberofbootstraps = 1
product_numbers = list(range(1624))
listofrowstotry = [1,2,3,4,5,6,7,8,9,10] 
nMinhashes = 1050

#instantiate the arrays that will store the performance measures results
PQ = [0] * numberofbootstraps
PC = [0] * numberofbootstraps
fractionOfComparisons = [0] * numberofbootstraps
averagePQ = [0] * len(listofrowstotry)
averagePC = [0] * len(listofrowstotry)
averageFractionOfComparisons = [0] * len(listofrowstotry)
TP = [0] * numberofbootstraps
FN = [0] * numberofbootstraps
FP = [0] * numberofbootstraps
averagePrecision = [0] * len(listofrowstotry)
averageRecall = [0] * len(listofrowstotry)
averageF1star = [0] * len(listofrowstotry)
averageF1 = [0]*len(listofrowstotry)
averageTP = [0] * len(listofrowstotry)
averageFN = [0] * len(listofrowstotry)
averageFP = [0] * len(listofrowstotry)

#LSH Performance 
for r in listofrowstotry:
  for j in range(numberofbootstraps):
    bootstrap = resample(product_numbers, replace=True, n_samples=len(product_numbers), random_state=1)
    Results = ProcedureLSHandMSM(data, nMinhashes, r, bootstrap)
    PQ[j]= Results[0] / Results[1] #duplicates found / number of comparisons
    PC[j] = Results[0] / Results[2] #duplicates found / total amount of duplicates present
    fractionOfComparisons[j] = Results[1]/ (((len(product_numbers)*len(product_numbers))-len(product_numbers))/2) #number of comparisons / maximum amount of comparisons 
    #MSM Performance
    twicetheamountofduplicatesfound =0
    twicetheamountofduplicatesWRONGLYfound =0
    ClusterMatrix = Results[3]
    print("cluster results:", ClusterMatrix ) #print the cluster matrix
    for row in range(len(ClusterMatrix)):
      for i in range(len(ClusterMatrix.iloc[row,1])):
        for j in range(len(ClusterMatrix.iloc[row,1])):
          if ClusterMatrix.iloc[row,1][i] == ClusterMatrix.iloc[row,1][j]:
            if i !=j:
              twicetheamountofduplicatesfound = twicetheamountofduplicatesfound+1
          elif ClusterMatrix.iloc[row,1][i] != ClusterMatrix.iloc[row,1][j]:
            twicetheamountofduplicatesWRONGLYfound = twicetheamountofduplicatesWRONGLYfound+1
    TP[j] = twicetheamountofduplicatesfound/2
    FN[j] = Results[2] - (twicetheamountofduplicatesfound/2)
    FP[j] = twicetheamountofduplicatesWRONGLYfound/2
  averageTP[listofrowstotry.index(r)] = sum(TP) / len(TP) #average TP for every number of rows used
  averageFN[listofrowstotry.index(r)] = sum(FN) / len(FN) #average FN for every number of rows used
  averageFP[listofrowstotry.index(r)] = sum(FP) / len(FP) #average FP for every number of rows used
  averagePQ[listofrowstotry.index(r)] = sum(PQ) / len(PQ) #average PQ for every number of rows used
  averagePC[listofrowstotry.index(r)] = sum(PC) / len(PC) #average PC for every number of rows used
  averageFractionOfComparisons[listofrowstotry.index(r)] = sum(fractionOfComparisons) / len(fractionOfComparisons) #average fraction of comparisons for every number of rows used

#calculate the average F1*, F1, Precision and Recall for every row number tried 
for r in range(len(listofrowstotry)):
  averageF1star[r] = (2 * averagePQ[r] * averagePC[r])/(averagePQ[r] + averagePC[r])

for r in range(len(listofrowstotry)):
  averagePrecision[r] = averageTP[r] / (averageTP[r] + averageFP[r])

for r in range(len(listofrowstotry)):
  averageRecall[r] = averageTP[r] / (averageFP[r] + averageFN[r])

for r in range(len(listofrowstotry)):
  averageF1[r] = (2 * averageRecall[r] * averagePrecision[r])/(averageRecall[r] + averagePrecision[r])

print("list of PQ averages: ", averagePQ)
print("list of PC averages: ", averagePC )
print("list of F1* averages:", averageF1star)
print("list of fraction of comparisons averages: ", averageFractionOfComparisons) 
print("list of TP averages:", averageTP)
print("list of FN averages:", averageFN)
print("list of FP averages:", averageFP)
print("list of Precision averages:", averagePrecision)
print("list of Recall averages:", averageRecall)
print("list of F1 averages:", averageF1)

#plot LSH PC Graph
# x = [averageFractionOfComparisons[0], averageFractionOfComparisons[1], averageFractionOfComparisons[2], averageFractionOfComparisons[3],averageFractionOfComparisons[4], averageFractionOfComparisons[5], averageFractionOfComparisons[6], averageFractionOfComparisons[7], averageFractionOfComparisons[8], averageFractionOfComparisons[9], averageFractionOfComparisons[10]]
# y = [averagePC[0], averagePC[1], averagePC[2], averagePC[3],averagePC[4], averagePC[5], averagePC[6], averagePC[7], averagePC[8], averagePC[9], averagePC[10]]
# plt.plot(x,y)
# plt.xlabel("Fraction of Comparisons")
# plt.ylabel("Pair Completeness")
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.show()

#plot LSH PQ Graph
# x = [averageFractionOfComparisons[0], averageFractionOfComparisons[1], averageFractionOfComparisons[2], averageFractionOfComparisons[3],averageFractionOfComparisons[4], averageFractionOfComparisons[5], averageFractionOfComparisons[6], averageFractionOfComparisons[7], averageFractionOfComparisons[8], averageFractionOfComparisons[9], averageFractionOfComparisons[10]]
# y = [averagePQ[0], averagePQ[1], averagePQ[2], averagePQ[3],averagePQ[4], averagePQ[5], averagePQ[6], averagePQ[7], averagePQ[8], averagePQ[9], averagePQ[10]]
# plt.plot(x,y)
# plt.xlabel("Fraction of Comparisons")
# plt.ylabel("Pair Quality")
# plt.xlim(0,0.2)
# plt.ylim(0,0.25)
# plt.show()

#plot LSH F1* Graph
# x = [averageFractionOfComparisons[0], averageFractionOfComparisons[1], averageFractionOfComparisons[2], averageFractionOfComparisons[3],averageFractionOfComparisons[4], averageFractionOfComparisons[5], averageFractionOfComparisons[6], averageFractionOfComparisons[7], averageFractionOfComparisons[8], averageFractionOfComparisons[9], averageFractionOfComparisons[10]]
# y = [averageF1star[0], averageF1star[1], averageF1star[2], averageF1star[3],averageF1star[4], averageF1star[5], averageF1star[6], averageF1star[7], averageF1star[8], averageF1star[9], averageF1star[10]]
# plt.plot(x,y)
# plt.xlabel("Fraction of Comparisons")
# plt.ylabel("F1*-Measure")
# plt.xlim(0,1)
# plt.ylim(0,25)
# plt.show()


#plot F1 Graph
# x = [averageFractionOfComparisons[0], averageFractionOfComparisons[1], averageFractionOfComparisons[2], averageFractionOfComparisons[3],averageFractionOfComparisons[4], averageFractionOfComparisons[5], averageFractionOfComparisons[6], averageFractionOfComparisons[7], averageFractionOfComparisons[8], averageFractionOfComparisons[9], averageFractionOfComparisons[10]]
# y = [averageF1[0], averageF1[1], averageF1[2], averageF1[3],averageF1[4], averageF1[5], averageF1[6], averageF1[7], averageF1[8], averageF1[9], averageF1[10]]
# plt.plot(x,y)
# plt.xlabel("Fraction of Comparisons")
# plt.ylabel("F1-Measure")
# plt.xlim(0,1)
# plt.ylim(0,0.5)
# plt.show()




        
      



    
