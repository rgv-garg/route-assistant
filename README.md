# Train delay predictor
 A train delay predictor trained with real data is a AI model that tries to predict the amount of time the train is delayed by from its expected arrival time.

**Features:-**

1) User can enter the train number and the station code along with the current date.
2) The model predicts the expected delay that might have caused keeping into account the historic delays, weather patterns, holiday patterns and maintenance work chances.
3) It displays the predicted delay which users can use to plan their travel.

**Data trained:-**

1) Historic data of the delays for a huge number of train numbers for each of the stations along the route.
2) Usage of recent delay data to test after the training.
3) Coordinates for all the stations with latitudes and longitudes.
4) Weather pattern (precipitaion, weather code, fog) for all the coordinates of these stations.
5) Accounting for the weekend and holiday delay surges.
6) Partially use three consecutive very high delay days to consider as a maintenance and incresed chance of delay on subsequent days
  
**Tech used:-**

1) Delta Lake
2) Unity Catalogue
3) DBFS vollumes
4) Apache spark
5) PySpark
6) PySpark SQL
7) Scikit Learn
8) Joblib
   
**Workflow:-**

<img width="560" height="602" alt="image" src="https://github.com/user-attachments/assets/a9284a4a-1c4a-402b-8a27-15506d937f44" />

**Sample UI:-**

<img width="1600" height="737" alt="image" src="https://github.com/user-attachments/assets/11645d4f-df9c-4700-9fad-c0b77cd198ea" />

**Comparision of our model prediction with the actual data for sample**

 Our prediction
<img width="725" height="578" alt="image" src="https://github.com/user-attachments/assets/4e58432f-3389-4822-bef5-93f4c7dc110c" />

Actual Data
<img width="1059" height="976" alt="image" src="https://github.com/user-attachments/assets/71a08db1-5b67-4a08-b3a1-2b1f3a728089" />

**Future extensions:-**

1) We have a model trained on the waitlist chances data for every train allowing users to predict the chance of securing a berth even before booking
2) We plan to train our train delay data with more data though current maintenance works in route specific tracks, and traffic light glitches
3) Expand the same model to metropolitan buses using real time traffic data
4) Crowd prediction at a particular station based on the tickets booked
5) Passenger rights chatbot
   


