#HiMICd-NCP: A high-resolution (daily and 1km) atmospheric moisture index collection over the North China Plain during 2003–2020

##1.Institutional information

1.1 Organization: School of Geography and Planning, and Guangdong Key Laboratory for Urbanization and Geo-simulation, Sun Yat-sen University, Guangzhou 510006, China.


##2.Contact information

2.1Authors:

Mrs. Zhiying Su (suzhy9@mail2.sysu.edu.cn)

Dr. Ming Luo (luom38@mail.sysu.edu.cn)


##3.Codes information

3.1Input Data Preprocessing.py

Gridded datasets processing, extraction values using weather station and data cleaning are recorded in this file. The section called “Data Acquisition” should be performed in Google Earth Engine, and the rest is operational under Python 3.10.

3.2HiMICd-NCP Model.py

The codes in “HiMICd-NCP Model.py” is used for model training and prediction, as well as for incorporating accuracy evaluation.

3.3Result Visualization.py

The overall accuracy assessment of the six indices and the accuracy visualization code are also publicly available in the file named “Result Visualization.py” and the sample data required is publicly available in the “Data Samples” folder.
