# route-assistant
Route Assistant trained with real data is a AI model that tries to predict the amount of time the train is delayed by from its expected arrival time.

Features:-
1) User can enter the train number and the station code along with the current date.
2) The model predicts the expected delay that might have caused keeping into account the historic delays, weather patterns, holiday patterns and maintenance work chances.
3) It displays the predicted delay which users can use to plan their travel.

Data trained:-
1) Historic data of the delays for a huge number of train numbers for each of the stations along the route.
2) Usage of recent delay data to test after the training.
3) Coordinates for all the stations with latitudes and longitudes.
4) Weather pattern (precipitaion, weather code, fog) for all the coordinates of these stations.
5) Accounting for the weekend and holiday delay surges.
6) Partially use three consecutive very high delay days to consider as a maintenance and incresed chance of delay on subsequent days
  
Tech used:-
1) Delta Lake
2) Unity Catalogue
3) DBFS vollumes
4) Apache spark
5) PySpark
6) PySpark SQL
7) Scikit Learn
8) Joblib

Workflow
