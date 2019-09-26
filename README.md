This project is about reading a geojson format and extracting information to pandas dataframe. After the initial cleaning of data, the data is written to a postgresql database and query optimization will be done for queries on column geo_Coord_lat , geo_coord_lon and crash_timestamp

# Few points worth mentioning regarding directory structure:
- json is placed under data/raw/
- Processed data is placed under data/processed after computation
- Notebook is present in notebooks folder
- traffic_crash is a package which load all initial packages(__init__.py) and sets data file paths under config.py
  - Abstracts the messy paths in the main notebook to here, so that cleaner view can be provided
- utils package is a general purpose utility package which contains API's for various purposes
- scripts folder is a placeholder for future purposes
## How to run this?
  -  start Anaconda prompt and go to this folder's parent location and type jupyter notebook
  - notebook starts with the parent folder as the base location and you can navigate to different projects from there



-------------------------------------------------- 
Exploratory data analysis
--------------------------------------------------
- Looks like longitude and latitude values are round off from geo_point_2d and geo_shape, So we can actually remove the redundant ones
- A lot of categorical data needs cleaning
	- crash_date (along with crash_time) needs to be converted to timestamp object and then stored
	- crash month, crash time are redundant, once we have all the data consolidated in timestamp object
	- Longitude and latitude can be stored as float data type, as we need to calculate distance using those
	- drv_age can be stored as int (for easier mathematical calculations) and later we can make it categorical if needed to analyze something
- I will be storing these longitude and latitude in geography data type(postgis) in psql database(for calculation ease)¶
