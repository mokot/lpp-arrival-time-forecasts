import gzip
import csv

import numpy as np
import scipy
import re
import matplotlib.pyplot as plt
from datetime import datetime
import holidays

from lpputils import parsedate, tsdiff, tsadd
from sklearn.linear_model import LinearRegression

PLOT_SAVE = True  # true if plot, false if save
HOLIDAYS = holidays.Slovenia()  # the holidays in Slovenia
RUSH_HOUR = [*range(6, 17)]  # the rush hour in Slovenia
# drivers and vehicles
NUM_OF_VEHICLES = None
NUM_OF_DRIVERS = None
NUM_OF_ROUTES = None
K = 25  # difference coefficient
MESTNI_LOG_VIZMARJE_PREDICTION = 150


def read_data(filename):
    """
    Reads the data from the given file and returns a tuple of two numpy arrays:
    the first one contains the legend, the second one the data.

    @param filename: the name of the file to read from
    @return: a tuple of two numpy arrays
    """
    file = gzip.open(filename, "rt", encoding="UTF-8")
    reader = list(csv.reader(file, delimiter="\t"))

    return np.asanyarray(reader[0]), np.asanyarray(reader[1:])


def write_data(filename, data, prediction):
    """
    Writes the data to the given file.

    @param filename: the name of the file to write to
    @param legend: the legend of the data
    @param data: the data to write
    @prediction: the prediction of the data
    @return: None
    """
    file = open(filename, "wt", encoding="UTF-8")

    [
        file.write(tsadd(temp_data, temp_prediction) + "\n")
        for temp_data, temp_prediction in zip(data[:, 6], prediction)
    ]

    return


def plot_graph(data_xy, title, file_name="graph.jpg"):
    """
    Plot data in 2d space.

    @param data_xy: the data to plot
    @param title: the title of the graph
    @param file_name: the name of the file to save the graph to
    @return: None
    """
    plt.figure(figsize=(17, 12))
    plt.suptitle(title, fontsize=25)

    # plot the data
    plt.scatter(range(len(data_xy)), data_xy[:, 1])

    plt.show() if PLOT_SAVE else plt.savefig(file_name)
    return


def calculate_mae(data, prediction):
    """
    Calculate the mean absolute error.

    @param data: the data to calculate the MAE from
    @param prediction: the prediction of the data
    @return: the MAE
    """
    # calculate the MAE (mean absolute error)
    return np.mean(np.abs(calculate_duration(data[:, 6], data[:, 8]) - prediction))


def get_matrix(data):
    """
    Get the matrix of the data.

    WEEK DAY (1-7) -> 0 = Monday, 1 = Tuesday, ..., 6 = Sunday
    MONTH (1-12) -> 0 = January, 1 = February, ..., 11 = December
    DAY (1-31) -> 0 = 1st, 1 = 2nd, ..., 30 = 30th, 31 = 31st
    HOUR (0-23) -> 0 = 0:00, 1 = 1:00, ..., 23 = 23:00
    MINUTE (0-59) -> 0 = 0:00, 1 = 0:01, ..., 59 = 0:59
    WEEKEND (0-1)
    HOLIDAY (0-1)
    SCHOOL HOLIDAY (0-1)
    AVERAGE DAILY TEMPERATURE (0-100)
    CLOUDINESS (0-100)
    STRING WIND (0-1)
    RAIN (0-1)
    SNOW (0-1)
    RUSH HOUR (0-1)
    VEHICLES (0-NUM_OF_VEHICLES)
    DRIVERS (0-NUM_OF_DRIVERS)
    ROUTES (0-NUM_OF_ROUTES)

    @param data: the data to get the matrix from
    @return: the matrix
    """
    return np.array(
        [
            # *get_week_day(data[:, 6]),
            # get_week_day_poly(data[:, 6]),
            # *get_month(data[:, 6]),
            # get_month_poly(data[:, 6]),
            *get_day(data[:, 6]),
            # get_day_poly(data[:, 6]),
            *get_hour(data[:, 6]),
            # get_hour_poly(data[:, 6]),
            *get_minute(data[:, 6]),
            # get_minute_poly(data[:, 6]),
            *get_time_week(data[:, 6]),
            *get_time_day(data[:, 6]),
            # get_weekend(data[:, 6]),
            get_holiday(data[:, 6]),
            get_school_holiday(data[:, 6]),
            *get_weather(data[:, 6]),
            *get_rush_hour(data[:, 6]),
            # *get_vehicles(data[:, 0]),
            # get_vehicles_poly(data[:, 0]),
            *get_drivers(data[:, 1]),
            # get_drivers_poly(data[:, 1]),
            # *get_route(data[:, 2])
        ]
    )


def filter_month(data, month, param=6):
    """
    Filter the data by month.

    @param data: the data to filter
    @param month: the month to filter by
    @param param: the parameter to filter by (default: 6)
    @return: the filtered data
    """
    return np.asanyarray(
        [date for date in data if parsedate(date[param]).month in month]
    )


def get_week_day(data):
    """
    Get the week day of the data.

    @param data: the data to get the week day from
    @return: the week day
    """
    week_day = np.zeros((len(data), 7))
    for i, date in enumerate(data):
        week_day[i, parsedate(date).weekday()] = 1
    return week_day.T


def get_week_day_poly(data):
    """
    Get the week day of the data.

    @param data: the data to get the week day from
    @return: the week day
    """
    return np.asanyarray([parsedate(date).weekday() for date in data])


def get_month(data):
    """
    Get the month of the data.

    @param data: the data to get the month from
    @return: the month
    """
    month = np.zeros((len(data), 12))
    for i, date in enumerate(data):
        month[i, parsedate(date).month - 1] = 1
    return month.T


def get_month_poly(data):
    """
    Get the month of the data.

    @param data: the data to get the month from
    @return: the month
    """
    return np.asanyarray([parsedate(date).month - 1 for date in data])


def get_day(data):
    """
    Get the day of the data.

    @param data: the data to get the day from
    @return: the day
    """
    day = np.zeros((len(data), 31))
    for i, date in enumerate(data):
        day[i, parsedate(date).day - 1] = 1
    return day.T


def get_day_poly(data):
    """
    Get the day of the data.

    @param data: the data to get the day from
    @return: the day
    """
    return np.asanyarray([parsedate(date).day - 1 for date in data])


def get_hour(data):
    """
    Get the hour of the data.

    @param data: the data to get the hour from
    @return: the hour
    """
    hour = np.zeros((len(data), 24))
    for i, date in enumerate(data):
        hour[i, parsedate(date).hour] = 1
    return hour.T


def get_hour_poly(data):
    """
    Get the hour of the data.

    @param data: the data to get the hour from
    @return: the hour
    """
    return np.asanyarray([parsedate(date).hour for date in data])


def get_minute(data):
    """
    Get the minute of the data.

    @param data: the data to get the minute from
    @return: the minute
    """
    minute = np.zeros((len(data), 60))
    for i, date in enumerate(data):
        minute[i, parsedate(date).minute] = 1
    return minute.T


def get_minute_poly(data):
    """
    Get the minute of the data.

    @param data: the data to get the minute from
    @return: the minute
    """
    return np.asanyarray([parsedate(date).minute for date in data])


def get_time_week(data, interval=3):
    """
    Get the time of the data.
    Split day into intervals of interval minutes

    @param data: the data to get the time from
    @interval: the interval of the time
    @return: the time
    """
    time = np.zeros((len(data), 7 * 24 // interval))
    for i, date in enumerate(data):
        time[i, (parsedate(date).weekday() * 24 + parsedate(date).hour) // interval] = 1
    return time.T


def get_time_day(data, interval=5):
    """
    Get the time of the data.
    Split day into intervals of interval minutes

    @param data: the data to get the time from
    @interval: the interval of the time
    @return: the time
    """
    time = np.zeros((len(data), 24 * 60 // interval))
    for i, date in enumerate(data):
        time[i, (parsedate(date).hour * 60 + parsedate(date).minute) // interval] = 1
    return time.T


def get_weekend(data):
    """
    Get the weekend of the data.

    @param data: the data to get the weekend from
    @return: 0 if day is weekend else 1
    """
    return np.asanyarray(
        [0 if (parsedate(date).weekday() in [5, 6]) else 1 for date in data]
    )


def get_holiday(data):
    """
    Get the holiday of the data.

    @param data: the data to get the holiday from
    @return: 0 if day is holiday else 1
    """
    return np.asanyarray([0 if (HOLIDAYS.get(date) != None) else 1 for date in data])


def get_school_holiday(data):
    """
    Get the school holiday of the data.

    @param data: the data to get the school holiday from
    @return: 0 if day is school holiday else 1
    """
    file = open("school_calendar_2012.txt", "r", encoding="UTF-8")
    school_holiday = list(re.split("\n", file.read()))[:-1]
    file.close()
    school_holiday = np.asanyarray(
        [datetime.strptime(date, "%d.%m.%Y").date() for date in school_holiday]
    )
    return np.asanyarray(
        [0 if (parsedate(date).date() in school_holiday) else 1 for date in data]
    )


def get_weather(data, temperature=5, cloudiness=65):
    """
    Get the weather of the data.

    @param data: the data to get the weather from
    @return: [daily temperature, cloudiness, strong wing, rain, snow]
    """
    file = open("weather.txt", "r", encoding="UTF-8")
    weather = np.asanyarray(
        [forecast.split(",")[2:] for forecast in list(re.split("\n\n", file.read()))][
            1:
        ]
    )
    file.close()

    # 0 = date
    # 1 = average daily temperature
    # 2 = cloudiness
    # 3 = strong wing
    # 4 = rain
    # 5 = snow
    return np.asanyarray(
        [
            [
                (
                    # 0 if float(forecast[1]) < temperature else 1,
                    # 0 if float(forecast[2]) > cloudiness else 1,
                    # 0 if forecast[3] == "da" else 1,
                    # 0 if forecast[4] == "da" else 1,
                    0
                    if forecast[5] == "da"
                    else 1,
                )
                for forecast in weather
                if date.startswith(forecast[0])
            ]
            for date in data
        ]
    )[:, 0].T


def get_rush_hour(data):
    """
    Get the rush hour of the data.

    @param data: the data to get the rush hour from
    @return: 0 if rush hour else 1
    """
    rush_hour = np.zeros((len(data), len(RUSH_HOUR)))
    for i, date in enumerate(data):
        if parsedate(date).hour in RUSH_HOUR and parsedate(date).weekday() < 5:
            rush_hour[i, RUSH_HOUR.index(parsedate(date).hour)] = 1
    return rush_hour.T


def get_vehicles(data):
    """
    Get the vehicles of the data.

    @param data: the data to get the vehicles from
    @return: the vehicles
    """
    # get the unique vehicles
    vehicles = np.zeros((len(data), NUM_OF_VEHICLES))
    for i, vehicle_id in enumerate(data):
        vehicles[i, int("".join(re.findall(r"\d", vehicle_id))) - 1] = 1
    return vehicles.T


def get_vehicles_poly(data):
    """
    Get the vehicles of the data.

    @param data: the data to get the vehicles from
    @return: the vehicles
    """
    # get the unique vehicles
    vehicle_ids = list(set((data)))
    return np.asanyarray([vehicle_ids.index(vehicle_id) for vehicle_id in data])


def get_drivers(data):
    """
    Get the drivers of the data.

    @param data: the data to get the drivers from
    @return: the drivers
    """
    # get the unique drivers
    drivers = np.zeros((len(data), NUM_OF_DRIVERS))
    for i, driver_id in enumerate(data):
        drivers[i, int(driver_id) - 1] = 1
    return drivers.T


def get_drivers_poly(data):
    """
    Get the drivers of the data.

    @param data: the data to get the drivers from
    @return: the drivers
    """
    # get the unique drivers
    driver_ids = list(set((data.astype(np.int32))))
    return np.asanyarray([driver_ids.index(int(driver_id)) for driver_id in data])


def get_route(data):
    """
    Get the routes of the data.

    @param data: the data to get the routes from
    @return: the routes
    """
    routes = np.zeros((len(data), NUM_OF_ROUTES))
    route_directions = np.empty(NUM_OF_ROUTES, dtype=object)
    for i, route_direction in enumerate(data):
        route_id = int(route_direction[0])

        # set the route direction
        if route_directions[route_id - 1] == None:
            route_directions[route_id - 1] = route_direction[1]

        routes[
            i,
            (
                route_id
                if (route_directions[route_id - 1] == route_direction[1])
                else (2 * route_id)
            )
            - 1,
        ] = 1

    return routes.T


def calculate_duration(departure, arrival):
    """
    Calculate the duration of the trip.

    @param departure: the departure time
    @param arrival: the arrival time
    @return: the duration
    """
    return np.asanyarray(
        [(tsdiff(arrival, departure)) for departure, arrival in zip(departure, arrival)]
    )


def get_model(model, train_data, test_data):
    """
    Genearate linear regression model and predict values.

    @param model: model key (route_id: route_direction)
    @param data: training data
    @param test_data: testing data
    @return: prediction for specific route
    """

    # filter data
    filter_train_data = []
    for temp_data in train_data:
        if temp_data[2] + ": " + temp_data[3] == model:
            filter_train_data.append(np.asanyarray(temp_data))
    filter_train_data = np.asanyarray(filter_train_data)
    filter_test_data = []
    for temp_data in test_data:
        if temp_data[2] + ": " + temp_data[3] == model:
            filter_test_data.append(np.asanyarray(temp_data))
    filter_test_data = np.asanyarray(filter_test_data)

    # if route is not in testing
    if not len(filter_test_data):
        return []

    # set number of vehicles and drivers
    global NUM_OF_VEHICLES
    NUM_OF_VEHICLES = np.max(
        [
            int("".join(re.findall(r"\d", vehicle_id)))
            for vehicle_id in np.concatenate(
                (filter_train_data[:, 0], filter_test_data[:, 0]), axis=0
            )
        ]
    )
    global NUM_OF_DRIVERS
    NUM_OF_DRIVERS = np.max(
        np.concatenate(
            (filter_train_data[:, 1], filter_test_data[:, 1]), axis=0
        ).astype(np.int32)
    )

    X = get_matrix(filter_train_data)

    # y = duration (arrival time - departure time)
    y = calculate_duration(filter_train_data[:, 6], filter_train_data[:, 8])

    # remove outliers (if duration is more than 1.5 hour)
    (outliers,) = np.where(y > 1.25 * 60 * 60)

    X = np.delete(X, outliers, axis=1)
    Xsp = scipy.sparse.csr_matrix(X).toarray()  # compressed sparse row matrix
    y = np.delete(y, outliers)

    # we want to predict the arrival time (duration)
    linear_regression = LinearRegression().fit(Xsp.T, y)
    # calculate the predicted data
    predicted_data = linear_regression.predict(get_matrix(filter_test_data).T)

    # special case is line with model = "1: MESTNI LOG - VIZMARJE"
    if "1" in model and "MESTNI LOG" in model and "VIŽMARJE" in model:
        predicted_data = [
            data + MESTNI_LOG_VIZMARJE_PREDICTION for data in predicted_data
        ]

    return predicted_data


if __name__ == "__main__":
    # read the data
    legend, train_data = read_data("train.csv.gz")
    # test_data = filter_month(train_data, month=[10])
    train_data = filter_month(train_data, month=[1, 2, 10, 11])  # filter by month
    _, test_data = read_data("test.csv.gz")

    predicted_dict = {}
    models_train = np.asanyarray(
        [temp_data[0] + ": " + temp_data[1] for temp_data in train_data[:, (2, 3)]]
    )
    for model in models_train:
        if not model in predicted_dict.keys():
            predicted_dict[model] = get_model(model, train_data, test_data)

    models_test = np.asanyarray(
        [temp_data[0] + ": " + temp_data[1] for temp_data in test_data[:, (2, 3)]]
    )
    predicted_data = []
    for model in models_test:
        predicted_data.append(predicted_dict[model].pop(0))
    predicted_data = np.asanyarray(predicted_data)

    # calculate the MAE
    # print("MAE:", calculate_mae(test_data, predicted_data))

    # write predicted data to file
    write_data("linear_regression.txt", test_data, predicted_data)
