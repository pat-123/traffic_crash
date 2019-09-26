from utils import *
#from sklearn.feature_selection import chi2
def print_(txt,flag = True):
    if flag:
        print(txt)
        
def print_missing_val_summary(full_data):
    for i,item in enumerate(full_data):
        mis = item.isnull().sum()
        txt = 'train' if i == 0 else 'test'
        if np.any(mis > 0) == False:
            print('No Missing values in ',txt)
        else:
            print('Missing values exist in',txt)
    
def get_missing_val_per(full_data,missing_values):
    #Get the missing percentages for both train and test
    for i,item in enumerate(full_data):
        # if i == 0:
        #    continue
        txt = 'Train' if i == 0 else 'Test'
        #get the rows with null values
        missing_values[txt] = pd.DataFrame(item.isnull().sum())
        #rename the columns old value: 0, new_value:'missing_per'
        missing_values[txt] = missing_values[txt].rename(columns = {'index':'variable', 0:'missing_per'})
        missing_values[txt]['missing_per'] = (missing_values[txt]['missing_per']/len(item))*100
    
    return missing_values

# Impute the missing values for training and test dataset
def impute_missing_val(full_data, except_=None):
    for i,dataset in enumerate(full_data):
        for col in full_data[i].columns:
            if col == except_:#skip it
                continue
            full_data[i].loc[:,col] = full_data[i].loc[:,col].fillna(full_data[i].loc[:,col].mean())#Replace with mean

def remove_sp(text):
    sp_ch = ['-']
    for sp in sp_ch:
        if type(text) is str:#type(text) == 'str':
           
            text = re.sub(sp,'',text)
    return text


def clean_num(data,series,col_name):
    """Takes a series as data along with the colname
    Returns the cleaned series column and removes special characters 
    and does datatype string to float"""
    
    #check values with - sign
    print('fare has these values with - sign')
    for index,row in enumerate(series):
        if type(row) is str  and '-' in row:
            print(row,':',index)
    #Remove - sign    
    for index,row in enumerate(series):
        #print(row)
        #if type(row) == 'str' and '-' in row:
        #    print('upd',row)
        txt = remove_sp(row)
        #print('git',txt)
        data.loc[index,col_name] =txt
        
    #data.fare_amount= data.fare_amount.astype('float')
    return data

def clean_coord(data):
    # lat,lon should be in range of -90 to 90, if not let us drop it
    data = data.drop(data[(data.pickup_latitude>90)|(data.pickup_latitude<-90)].index)
    data = data.drop(data[(data.pickup_longitude>90)|(data.pickup_longitude<-90)].index)
    data = data.drop(data[(data.dropoff_latitude>90)|(data.dropoff_latitude<-90)].index)
    data = data.drop(data[(data.dropoff_longitude>90)|(data.dropoff_longitude<-90)].index)
    data = data.reset_index().drop('index',axis=1)
    return data

def calculate_dis(full_data):
    for i,dataset in enumerate(full_data):
        for rowi in (dataset.index):
            #print(rowi)
            p1 = (dataset.pickup_latitude[rowi],dataset.pickup_longitude[rowi])
            p2 = (dataset.dropoff_latitude[rowi],dataset.dropoff_longitude[rowi])
            x1 = float(p1[0])
            y1 = float(p1[1])
            x2 = float(p2[0])
            y2 = float(p2[1])
            if (p1 == 40.6438) | (p2 == -73.7823):
                printf('It is ride to/from airport at %s'%rowi)
                dataset.loc[rowi,'remarks'] += '-airport'
            dataset.loc[rowi,'mahattan_dis'] = abs((x1-x2)) + abs((y1-y2))
            dataset.loc[rowi,'euclid_dis'] = np.sqrt( pow((x1-x2),2) + pow( (y1-y2),2) )
            dataset.loc[rowi,'geodesic'] = geodesic(p1,p2).miles
        #if i ==0:
        #    data=full_data[0]
        #else:
        #    test=full_data[1]
    return full_data

def parsedate(dataset, col_name):
    print(dataset)
    for index, row in enumerate(dataset.loc[:,col_name]):
        if '-' in row:
            print(index,row,type(row))
            dataset.loc[index,col_name] = pd.Timestamp(' '.join( dataset.loc[index,col_name].split()[0:2]))
            print(dataset.loc[index,col_name].dtype)
        else:
            print_('Invalid pickup time for %s'%index)
            dataset.loc[index,col_name] = dataset.loc[index-1,col_name]#replace with neighbours value
    return dataset

#Functions block
def get_fn_ratio(dv):
    return len(dv[dv == 0]),len(dv[dv == 1])

def best_fxn(skew_kurt_dic):
    for dataset in skew_kurt_dic.keys():
        for col in skew_kurt_dic[dataset].keys():
            print(skew_kurt_dic[dataset][col].keys())
            
def skew_kurt_analysis(data,label_,col, skew_kurt_dic,printhist = False,printverbose=True):
    if printverbose:
        print('---------------%s---------------'%label_)
    if(printhist):
        print(data.hist(label = label_))
    skew_ =  skew(data)
    
    if printverbose:
        print('Skewness : ',skew_)
    if abs(skew_) < 0.5 :
        txt = '%%%%%Moderately symm'
    else:
        txt = '+++++very %s skewed'%('Right/positive(tail/majority towards right)' if skew_ > 0 else 'Left/negative(tail/majority towards left)')
    if printverbose:
        print(txt)
    kurtosis_ =  kurtosis(data)
    if printverbose:
        print('Kurtosis : ',kurtosis_)
    if abs(kurtosis_) < 0.5 :
        txt = '%%%%%%%%%Moderately Bell shaped'
    else:
        txt = '^^^^^^^^Flat' if kurtosis_ >0 else '~~~~~~~~~~Pointy'
    if printverbose:
        print(txt)
    skew_kurt_dic[col][label_] = (skew_,kurtosis_)
    #skew_kurt_dic[col]['maxlabel'] = ''
    return skew_kurt_dic

def key_withmax_value(dic):
    max = -1
    for key in dic.keys():
        if max <  dic[key]:
            max = dic[key]
    #print(max)
    inv_map = {v:k for k,v in dic.items()}
    #print(inv_map)
    return inv_map[max]

def key_withmin_value(dic):
    min = 1000
    for key in dic.keys():
        if min >  dic[key]:
            min = dic[key]
    #print(max)
    inv_map = {v:k for k,v in dic.items()}
    #print(inv_map)
    return inv_map[min]

def maxlabel(dic):
    label_ = 'original'
    for label in dic.keys():
        if label == 'maxlabel':
            continue
        
        if abs(dic[label][0]) <= 0.5 and abs(dic[label][1]) <=0.5:
            label_ = label
            break
    return label_

def best_fxn(skew_kurt_dic):
    #maxlabel in each column key represents the function which gives distribution closest to normal distribution
    for col in skew_kurt_dic.keys():
            skew_kurt_dic[col]['maxlabel'] = maxlabel(skew_kurt_dic[col])

 

def iv_dv_get_col_names(data,target):
    continous_names = []
    categ_names = []
    for i in data.columns:
        #if i == target:
        #    continue
        if(data.loc[:,i].dtype not in [str,object]) and len(data.loc[:,i].value_counts()) <30:
            print('************************you may need to define this col - %s in categ variable'%i)
            print('******* adding %s(%s) as continous as per the data type'%(i,data.loc[:,i].dtype))
            continous_names.append(i)
        elif data.loc[:,i].dtypes in ['int64','int32','float32','float64']:
            print('$$ adding %s(%s) as continous'%(i,data.loc[:,i].dtype))
            continous_names.append(i)  
        else:
            print('$$ adding %s(%s) as categorical'%(i,data.loc[:,i].dtype))
            categ_names.append(i)
    iv = data.loc[:,data.columns!=target]# data.drop(['y'],axis=1)
    dv = data.loc[:,target]
    iv_train, iv_test, dv_train,dv_test = train_test_split(iv,dv, test_size = 0.2,random_state=42)
    return continous_names,categ_names, iv, dv,[iv_train, iv_test, dv_train,dv_test]


def iv_dv_get_col_names_dist_based(data,target):
    """This function tries to see the value counts of each of the unique values and estimate if we have any discrete
    variable with numeric datatype. If so we would like to segregate that col to categorocal variable and run chi-sq analysis
    , so that better insight can be achieved"""
    
    #for col in data.columns:
    #   if data.loc[:,col].dtype is not str and len(data.loc[:,col].value_counts()) <30:
    #        print('Need to define this col - %s in categ variable'%col)
    
    return iv_dv_get_col_names(data,target)#distinguish based on data types

def get_col_names(data):#kept this for backward compatibility
    continous_names = []
    categ_names = []
    for i in data.columns:
        continous_names.append(i) if data.loc[:,i].dtypes in ['int64','int32','float32','float64'] \
        else categ_names.append(i)
    iv = data.loc[:,data.columns!=target]# data.drop(['y'],axis=1)
    dv = data.loc[:,target]
    iv_train, iv_test, dv_train,dv_test = train_test_split(iv,dv, test_size = 0.2,random_state=42)
    return continous_names,categ_names

def plot_dist(var,plot=True):
    
    m = var.mean()
    sd = var.std()
    if plot == True:
        sb.distplot(var)
        plt.plot([m,m,m],[0,1,2],'r',label = 'mean')
        plt.plot([m+sd,m+sd,m+sd],[0,1,2],'g',label = '68%')
        plt.plot([m-sd,m-sd,m-sd],[0,1,2],'g')
        plt.plot([m+2*sd,m+2*sd,m+2*sd],[0,1,2],'y',label = '95%')
        plt.plot([m-2*sd,m-2*sd,m-2*sd],[0,1,2],'y')
        plt.legend()
    return m-2*sd,m-sd,m+sd,m+2*sd

def z_score(data,target):
    confidence_interval = {}
    for col in data.columns:
        if col == target:
            continue
        
        x = data.loc[:,col]
        std_error = x.std()/np.sqrt(len(data))
        #print(col,x.mean(),x.std())
        cum_z_score = t.ppf((1+0.95)/2., n-1)
        #print(z)
        confidence_interval[col] = [x.mean()+cum_z_score*std_error, x.mean()-cum_z_score*std_error]
    return confidence_interval

def distplot_all(data,colnames):
    plotno=321
    for colname in colnames:
        plt.figure(figsize=(10,10))
        plt.subplot(plotno)
        plotno+=1
        var = '_%s'%plotno
        var = sb.distplot(data.loc[:,colname])
    plt.show()
    
def distribution_summary(data_slice, col_name,show=False):
    # Get skewness and Kurtosis to analyze the shape of distribution for var
    var = data_slice
    col = col_name
    skew_kurt_dic={}
    skew_kurt_dic[col]={}
    skew_kurt_dic = skew_kurt_analysis((var),'original',col, skew_kurt_dic,show)
    if(not (len(data_slice[data_slice == 0]) > 0) ):#If no zero value
        show_=True
        skew_kurt_dic = skew_kurt_analysis(np.log(var),'log',col,skew_kurt_dic,show)
    else:
        skew_kurt_dic['log'] =(-1,-1)
        show_ = False
    skew_kurt_dic = skew_kurt_analysis(np.sqrt(var),'sqrt',col,skew_kurt_dic,show)
    skew_kurt_dic = skew_kurt_analysis(np.cbrt(var),'cbrt',col,skew_kurt_dic,show)

    if show_:
        plt.legend(loc='upper right')
    plt.xlabel(col)
    plt.savefig('../data/processed/skew_anal_%s'%col,bbox_inches='tight')
    
    
 # Impute the missing values for training and test dataset
def impute_missing_val(full_data, except_=None):
    for i,dataset in enumerate(full_data):
        for col in full_data[i].columns:
            if col == except_:#skip it
                continue
            full_data[i].loc[:,col] = full_data[i].loc[:,col].fillna(full_data[i].loc[:,col].mean())#Replace with mean
            
def get_outliers(data,continous_names, outliers_,except_ = [],drop=False):
    #continous_names = [ x for x in data.columns if x not in [except]]
    
    for i in continous_names:
        outliers_[i] = []
        q75,q25 = np.percentile(data.loc[:,i],[75,25])
        iqr = q75 - q25
        min = q25 - (iqr* 1.5)#lower fence
        max = q75 + (iqr* 1.5)#upper fence


        print('-----values below %s and above %s %s for col %s:'%(min,max,'dropped ' if drop else 'suggested to drop',i))  
        #TBD: can fill actual faulter rows in outliers_
        outliers_[i].append('per outliers above max for %s is %s'%(i, ( (len(data[data.loc[:,i] > max])/len(data)) )*100 ))
        outliers_[i].append('per outliers below min for %s is %s'%(i, ( (len(data[data.loc[:,i] < min])/len(data)) )*100 ))
        if drop:
            data = data.drop(data[data.loc[:,i] <min].index)
            data = data.drop(data[data.loc[:,i] > max].index)
            data = data.reset_index().drop('index',axis=1)
    return outliers_,data
  
def feature_eng(full_data):
    for i,dataset in enumerate(full_data):
        for rowi in range(dataset.shape[0]):
            #print(dataset.pickup_datetime[rowi],dataset.pickup_datetime[rowi].is_month_end)
            if dataset.pickup_datetime[rowi].date().day in [26,27,28,29,30,31]:
                dataset.loc[rowi,'remarks'] = 'month_end'
            elif dataset.pickup_datetime[rowi].is_month_start:
                dataset.loc[rowi,'remarks'] = 'month_start'
            elif dataset.pickup_datetime[rowi].is_year_end:
                dataset.loc[rowi,'remarks'] = 'year_end'
            elif dataset.pickup_datetime[rowi].is_year_start:
                dataset.loc[rowi,'remarks'] = 'year_start'
            elif not dataset.pickup_datetime[rowi].isoweekday:
                dataset.loc[rowi,'remarks'] = 'weekend'

            else:
                dataset.loc[rowi,'remarks'] = 'normal'
            if dataset.pickup_datetime[rowi].hour in [17,18,19,20,21]:#peak times
                #dataset.loc[rowi,'hour'] = 'evening'
                dataset.loc[rowi,'time'] = 'peakeveningtime' 
            elif dataset.pickup_datetime[rowi].hour in [22,23,0,1,2,3,4]:#peak times
                dataset.loc[rowi,'time'] = 'peaknighttime' 
                #dataset.loc[rowi,'hour'] = 'night'
            elif dataset.pickup_datetime[rowi].hour in [11,12,13,14,15,16]:
                dataset.loc[rowi,'time'] = 'noontime' 
                #dataset.loc[rowi,'hour'] = 'noon'
            else:
                dataset.loc[rowi,'time'] = 'regular'
                #dataset.loc[rowi,'hour'] = 'regular'
            date_ =dataset.pickup_datetime[rowi].date()
            dataset.loc[rowi,'day'] = int(date_.day)
            dataset.loc[rowi,'month']  = int(date_.month)
            dataset.loc[rowi,'year'] = date_.year
            dataset.loc[rowi,'weekday']=int(dataset.pickup_datetime[rowi].weekday())
            
            #dataset.loc[rowi,'hour'] = int(dataset.pickup_datetime[rowi].hour)
            #m.date().day,m.date().month,m.date().year
        dataset.loc[:,'season'] = dataset.pickup_datetime.apply(lambda x:x.month).apply(lambda x:'winter' \
                                                    if x in [11,12,0,1,2] else ('summer' if x in [5,6,7]\
                                                                    else ('spring' if x in [3,4] else 'fall')))
    return full_data  

def parsedate(dataset, col_name):
    for index, row in enumerate(dataset.loc[:,col_name]):
        if '-' in row:
            #print(index,row,type(row))
            dataset.loc[index,col_name] = pd.Timestamp(' '.join( dataset.loc[index,col_name].split()[0:2]))
        else:
            print_('Invalid pickup time for %s'%index)
            dataset.loc[index,col_name] = dataset.loc[index-1,col_name]#replace with neighbours value
    return dataset


def chi_sq_analysis(data,cat_names,target,drop=False):
    skip_list = []
    print('Deleted skip col below-------------' if drop else 'Suggest to skip col below---------------------')
    for i in cat_names:
        chi2, p, dof,ex = chi2_contingency(pd.crosstab(data.loc[:,target], data[i]))
        if(p<0.05):#Null hypothesis rejected
            print('accept -pvalue %f %s'%(p,i))

        else:#Fail to reject Null hypothesis
            print('skip -p value %f %s'%(p,i))
            skip_list.append(i)
    if drop:
        data = data.drop(skip_list,axis=1)
    
    
    #Remove irrelevant categorical variables
    cat_names = [ x for x in cat_names if x not in skip_list]
    
    return data,skip_list


def fitols(iv,dv,intercept = True):
    model = sm.OLS(dv,sm.add_constant(iv)).fit() if intercept else sm.OLS(dv,iv).fit()
    #An intercept is not included by default and should be added by the user. See statsmodels.tools.add_constant.
    return model.summary()

def vif_summary(data,target,manual=False):
    if manual:
        THR = 80
        skip_vif = []
        vif_val = []
        
        #remove dependent variable for vif analysis
        cnames = [col for col in data.columns if col not in [target]]
        tmp = data[cnames]
        for i in range(len(cnames)):
            #print('goin for',cnames[i])
            dv = data.loc[:,tmp.columns==cnames[i]]
            iv = sm.add_constant(tmp.loc[:,tmp.columns!=cnames[i]])
            #print(iv,dv)
            model = sm.OLS(dv,iv).fit()
            r2 = model.rsquared
            vif = 1/(1-r2)

            if round(vif,2) > THR:
                
                skip_vif.append(cnames[i])
            print("%s r square : %s vif %s "%(cnames[i], r2, vif))
            vif_val.append(vif)
        return pd.Series(vif_val,cnames)
    else:#use in-built
        vif = pd.DataFrame()
        predictors = model_data_orig.drop(['fare_amount'],axis=1)
        vif["VIF"] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]
        vif["features"] = predictors.columns
        return vif
    
def r2(val):
    return 1/(1-val)

def cmp_model(iv_,dv_):
    print('LinearRegression: r2',LinearRegression().fit(iv_,dv_).score(iv_,dv_),'vif :',r2(LinearRegression().fit(iv_,dv_).score(iv_,dv_)))
    print('ols: r2',sm.OLS(dv_,sm.add_constant(iv_)).fit().rsquared,'vif :',r2(LinearRegression().fit(iv_,dv_).score(iv_,dv_)))
    return sm.OLS(dv_,sm.add_constant(iv_)).fit().summary()

def normalize(data,colnames):
    for i in colnames:
        data[i] = (data[i] - data[i].min())/(data[i].max() - data[i].min())
    return data

def model_score(actual_, pred_):
    
    actual = actual_.values
    pred = pred_
    #print(actual,pred)
    ssr = 0.0
    sstotal = 0.0
    for i in range(len(actual)):
        ssr = ssr +(pow(actual[i] - pred[i],2))
        sstotal += (pow(actual[i] - actual.mean(),2))
    r2 = 1 - (ssr/sstotal)
    return r2,np.sqrt(mean_squared_error(actual,pred))
#---------------------Model building

def get_score(data,clf, iv, dv,regression=False):
    i = 0
    sum_score = 0
    it_score = 0
    sum_auc = 0
    sum_rmse = 0
    count_auc = 0
    count_rmse=0
    KFOLDS = 5
    logger = {}
    kf = KFold(n_splits = KFOLDS)
    for train_index,test_index in kf.split(range(data.shape[0])):
        iv_train, iv_test, dv_train,dv_test = iv[train_index], iv[test_index]\
                                ,dv[train_index], dv[test_index]
        #print(iv_train[0:2] ,iv_test[0:2])
        clf.train(iv_train, dv_train)
        dv_test_pred = clf.predict(iv_test)
        it_score = clf.score(iv_test, dv_test)
        sum_score += it_score
        print('model score :',it_score)
        
        #To be updated later
        
        if not regression:
            auc_ = clf.auc_model(iv_test,dv_test)
            if np.isnan(auc_) != True:
                sum_auc += auc_
                print('auc :',auc_)
                count_auc += 1
            else:
                print('fpr is nan')
            #auc_for_model.append([auc])
        
        if regression:#add rmse score also
            rmse_ = np.sqrt(mean_squared_error(dv_test,dv_test_pred))
            print('rmse :',rmse_)
            if it_score <0:
                print('~~~~~check logger, r2 is negative')
                logger['iv_train'] = iv_train
                logger['iv_test'] = iv_test
                logger['dv_train'] = dv_train
                logger['dv_test'] = dv_test
            else:
                sum_rmse += rmse_
                count_rmse +=1
            

    return (sum_score/KFOLDS, sum_auc/count_auc if not regression else -1,sum_rmse/count_rmse if count_rmse else -1,logger)



class sklearnHelper(object):
    def __init__(self, clf, seed = 0, params = None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()
    def train(self, train_iv, train_dv):
        self.clf.fit(train_iv, train_dv)#train on training data
    def predict(self, test_iv):
        return self.clf.predict(test_iv)
    def score(self, iv_test, dv_test):
        return self.clf.score(iv_test, dv_test)
    def feature_importances(self,x, y):
        print(self.clf.fit(x,y).feature_importances_)
    def auc_model(self,iv_test,dv_test):
        dv_test_proba = self.clf.predict_proba(iv_test)[::,1] 
        
        fpr,tpr,threshold = roc_curve(dv_test,dv_test_proba)
        
        auc_ = auc(fpr,tpr)
        #if np.isnan(auc_):
        #    print(fpr,tpr)
        return auc_


def plot_rf_feature_importance(iv,dv,rf_params):
    iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.2,random_state=42)
    model = RandomForestRegressor(**rf_params).fit(iv_train,dv_train)
    print('accuracy ' ,model.score(iv_test,dv_test))
    feature_dataframe = pd.DataFrame( {'features': iv_train.columns,
         'Random Forest feature importances': model.feature_importances_})

    f,ax = plt.subplots(figsize=(25,5))
    plt.plot(list(feature_dataframe.features),list(feature_dataframe['Random Forest feature importances']))
    plt.ylabel('output')
    
def rf_hyperparameter_tuning(iv,dv,rf_params):
    print('--------------------random forest pramaeter tuning------------------')
    rf_params = {}
    #-------- no of trees
    ntrees = [100,200,500,1000]
    score_={}
    iv_train,iv_test,dv_train,dv_test =train_test_split(iv,dv,test_size=0.2,random_state=42)
    for nt in ntrees:
        model = RandomForestRegressor(n_estimators=nt,random_state=42).fit(iv_train,dv_train)
        score_[nt]=model.score(iv_test,dv_test)
    #print(score_)
    plt.plot(score_.keys(),score_.values())
    plt.ylabel('score')
    plt.xlabel('no of trees')
    plt.show()
    rf_params['n_estimators'] = key_withmax_value(score_)
    #print(rf_params['n_estimators'])
    #-------- no of max features
    score_={}
    max_feat = ['log2','sqrt','auto',int(np.ceil(iv_train.shape[1]/3))]
    for m in max_feat:
        model = RandomForestRegressor(n_estimators=rf_params['n_estimators'],max_features=m,random_state=42).\
        fit(iv_train,dv_train)
        score_[m]=model.score(iv_test,dv_test) 
    #print(score_)
    plt.plot(score_.keys(),score_.values())
    plt.ylabel('score')
    plt.xlabel('no of maximum features')
    plt.show()
    rf_params['max_features'] = key_withmax_value(score_)
    rf_params['random_state'] = 42
    return rf_params

def knn_k_tuning(iv,dv,knn_params):
    print('-------------------------------KNN neighbors tuning---------------')
    iv_train,iv_test,dv_train,dv_test =train_test_split(iv,dv,test_size=0.2,random_state=42)
    rmse_err = {}
    knn_idle = np.sqrt(iv.shape[0]).astype(int)#sqrt of no of observations
    print('Idle sqrt value for neigbors is',knn_idle)
    for k in list(range(1,21))+[knn_idle-3,knn_idle-2,knn_idle,knn_idle+2,knn_idle+3]:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(iv_train,dv_train)
        dv_test_pred = model.predict(iv_test)
        err =np.sqrt(mean_squared_error(dv_test,dv_test_pred))
        rmse_err[k] = err
        print('rmse error for k %s is %s'%(k,err))
    print(rmse_err)
    knn_params['n_neighbors'] = key_withmax_value(rmse_err)
    plt.plot(rmse_err.keys(),rmse_err.values())
    plt.ylabel('rmse')
    plt.xlabel('k value')
    plt.show()
    return knn_params
    
def chart_regression(dv_test,dv_test_pred, sort=True):
    t = pd.DataFrame({'pred': dv_test_pred, 'actual': dv_test})
    plotno=321
    if sort:
        t.sort_values(by=['actual'], inplace=True)
    f,ax=plt.subplots(figsize=(20,10))
    plt.subplot(plotno+2)
    plt.plot(t['actual'].tolist(), label='actual')
    #plt.scatter(t['actual'].tolist())
    plt.plot(t['pred'].tolist(), label='predicted')
    plt.ylabel('output')
    plt.legend()#shows x,y labels set above
   

def regression_metrics(actual, predicted):
    #print(actual)
    if not np.any(actual,0):
        mape_ = np.mean(np.abs(actual-predicted)/actual)*100
    else:#calculate manually
        rmse=0
        count=0
        k=0
        for i,j in zip(actual,predicted):
            if i==0:
                print('there are a few zeroes in actual data')
            else:
                count+=1
                k+=(abs((i-j)/i))
                #print(i,j,k)q`
                rmse+=(pow((i-j),2))
                
        mape_ = (k/count)*100
    accuracy = 100-mape_#accuracy
    rmse = np.sqrt(mean_squared_error(actual,predicted))
    chart_regression(actual,predicted)
    print('+++++++++accuracy : %s'%accuracy)
    print('----------MAPE : %s',mape_)
    print('---------RMSE %s',rmse)

                   

def plot_lr_coef(iv,dv):
    
    #iv.shape,dv
    plotno=321
    iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.2,random_state=42)
    #model=RandomForestRegressor(n_estimators=10).fit(iv_train,dv_train)
    model=LinearRegression(copy_X= True, fit_intercept=True).fit(iv_train,dv_train)
    dv_test_pred = model.predict(iv_test)
    
    dv_train_pred=model.predict(iv_train)
    print_score(model,dv_test, dv_test_pred, dv_train, dv_train_pred,iv_test)
    
    f,ax=plt.subplots(figsize=(20,20))
    plt.subplot(plotno)
    plt.plot(model.coef_,iv_train.columns)
    plt.xlabel('coefficent')
    plt.ylabel('predictors')
    print(iv_train.columns,model.coef_)
    print('LR R2 score :',model.score(iv_test,dv_test))
    n = iv.shape[0]
    k = iv.shape[1]
    r2,rmse = model_score(dv_test,dv_test_pred)
    if r2>0:
        adj_score = 1-( ((1-r2)*(n-1))/(n-k-1) )
        print('Adjusted R2 Score :',adj_score)
    regression_metrics(dv_test,dv_test_pred)
    plt.show()

def boxcox_normalization(data,norm_col_data,test,norm_col_test):
    #norm_col = ['euclid_dis','month','day']
    norm_data1 = data.drop(norm_col_data,axis=1)#except the normalized columns
    norm_test1 = test.drop(norm_col_test,axis=1)#except the normalized columns
    df_data = data[norm_col_data]
    df_test = test[norm_col_test]
    df_data_after = pd.DataFrame()
    df_test_after = pd.DataFrame()
    fitted_lambdas={}
    for col in df_data.columns:
        print(col)
        df_data_after[col],fitted_lambda = stats.boxcox(df_data.loc[:,col].values)
        fitted_lambdas[col] = fitted_lambda#store for predicting the test later
        if col in norm_col_test:
            print('Applying %s to %s in test'%(fitted_lambda,col))
            df_test_after[col] = stats.boxcox(df_test.loc[:,col],fitted_lambda)
            
    df_data_normalized = pd.concat([norm_data1,df_data_after],axis=1)
    df_test_normalized = pd.concat([norm_test1,df_test_after],axis=1)
    #print(df_data_normalized.columns)
    return df_data_normalized,df_test_normalized,fitted_lambdas
def mape(actual, predicted):
    mape_ = np.mean(np.abs(actual-predicted)/actual)
    return mape_*100

def rmse(actual, predicted):
    rmse_ = np.sqrt(mean_squared_error(actual,predicted))
    return rmse_


def print_score(model,dv_test, dv_test_pred, dv_train, dv_train_pred,iv_test):
    print('accuracy',100-mape(dv_test, dv_test_pred))#accuracy
    print('model score :',model.score(iv_test ,dv_test))
    print('model r2 score',r2_score(dv_test,dv_test_pred))
    test_deviation = rmse(dv_test, dv_test_pred)
    train_deviation = rmse(dv_train, dv_train_pred)
    print('test: rmse',test_deviation)#deviation of test data
    #print(rmse(dv_train, dv_train))#no deviation
    print('train:rmse',train_deviation)#deviation of train data


    if(test_deviation - train_deviation < 1 ):
        print ('%%%%%%%%%%Good model')
    else :
        if(test_deviation > train_deviation ):
            print('XXXXXXX overfitting problem')
        else:
             print('---------underfitting problem')
#-----------------specific
def plot_rides(sub_data,title_='scatterplot'):
    plt.scatter(sub_data.mahattan_dis, sub_data.fare_amount,label='distance')
    plt.scatter(sub_data.passenger_count, sub_data.fare_amount,color='red',label='passenger_count')
    plt.legend()
    plt.ylabel('fare')
    plt.title(title_)
    plt.show()

def variation(high_fare,txt):
    plotno=321
    plt.figure(figsize=(15,11))
    plt.subplot(plotno)
    plt.title(txt)
    var = '_%s'%plotno
    high_fare.groupby('passenger_count').fare_amount.mean().plot.bar()
    plt.ylabel('average fare')
    plotno+=1
    plt.subplot(plotno)
    high_fare.groupby('year').fare_amount.mean().plot.bar()
    plt.ylabel('average fare')
    plotno+=1
    plt.subplot(plotno)
    var = high_fare.groupby('month').fare_amount.mean().plot.bar()
    plt.ylabel('average fare')
    plotno+=1
    plt.subplot(plotno)
    high_fare.groupby('day').fare_amount.mean().plot.bar()
    plt.ylabel('average fare')
    plotno+=1
    plt.subplot(plotno)
    high_fare.groupby('time').fare_amount.mean().plot.bar()
    plt.ylabel('average fare')
    plotno+=1
    plt.subplot(plotno)
    high_fare.groupby('season').fare_amount.mean().plot.bar()
    plt.ylabel('average fare')
    plt.show()

    
def plot_rides_time(data,date_=False,hour_=False):
    hour = []
    fare=[]
    date =[]
    for i in range(len(data)):
        if date_:
            date.append(data.pickup_datetime.iloc[i].date())
        else:
            hour.append(data.pickup_datetime.iloc[i].hour)
        fare.append(data.fare_amount.iloc[i])
    if hour_:
        plt.scatter(hour,fare)
        plt.xlabel('pickup hour')
    else:
        plt.scatter(date,fare)
        plt.xlabel('pickup year')
    plt.ylabel('fare')
    
