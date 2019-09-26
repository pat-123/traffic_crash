from utils import *

#-----------------specific
def dist(x1,y1,x2,y2):
    #for rowi in data.index:
    #p1 = (data.loc[rowi,'geo_coord_lat'],data.loc[rowi,'geo_coord_lon'])
    #p2 = (def_lat,def_lon)
    #x1 = float(p1[0])
    #y1 = float(p1[1])
    #x2 = float(p2[0])
    #y2 = float(p2[1])
    p1 = (x1,y1)
    p2 = (x2,y2)
    print('dist b/w (%s,%s) and (%s,%s) is %s'%(x1,y1,x2,y2,geodesic(p1,p2)))
    return geodesic(p1,p2).km
    
#print(dist(data.iloc[ind,:], 35.95956710652295,-78.95570271414434))


def join(data,rowi):
    
    val = ''
    for col in data.columns:
        if data.loc[:,col].dtype ==object:
            if col == 'crash_timestamp':
                val += '\'%s\','%str(data.loc[rowi,col])
            else:
                val += '\'%s\','%data.loc[rowi,col] 
        else:
            val += '%s,'%data.loc[rowi,col] 
    return val[:-1]    


#####-----------------------Database connect
hostname = 'localhost'
username = 'postgres'
password = 'calvin'
database = 'traffic_crash'
myConnection = psycopg2.connect( host=hostname, user=username, password=password, database=database )

def connect():
    cursor = myConnection.cursor()
    return cursor

def exec(cursor, query):
    cursor.execute(query)
    myConnection.commit()

def close():
    myConnection.close()  
