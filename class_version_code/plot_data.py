import matplotlib.pyplot as plt


class PlotData:
    """
    A class used to plot forecast data

    Attributes
    ----------
    plot_size : int
        a number that dictates how many training data-points
        before the forecasting line are going to be in the plot
    train : Series
        a data series that was used to train the forecasting method/model
    test : Series
        a data series that was used to test the result of the forecasting method/model
    predictions : list
        a list of predictions that are the result of the trained forecasting method/model
    """

    def __init__(self, plot_size, train, test, predictions):
        self.plot_size = plot_size
        self.train = train
        self.test = test
        self.predictions = predictions

    def plot_data(self, title):
        """
        Plots the real data-points as well as the predictions from all of
        the forecasting methods/models

        Parameters
        ----------
        title : str
            title of the plot
        """
        # take only the last "self.plot_size" number of points to show on plot
        train = self.train[-self.plot_size:]
        fig, ax = plt.subplots(figsize=(12, 6))
        # real data is plotted in grey
        ax.plot(self.test.index, self.test.values, color="gray")
        ax.plot(train.index, train.values, color='gray')
        # predictions are plotted in different colors
        ax.plot(self.test.index, self.predictions[0], label="holts method", color='blue')
        ax.plot(self.test.index, self.predictions[1], label="ARIMA model", color='orange')
        ax.plot(self.test.index, self.predictions[2], label="LSTM model", color='green')
        # create a line that tells us when the forecasting starts
        ax.vlines(train.index[-1], 0, max(list(self.train.values) + list(self.test.values)), linestyle='--', color='r',
                  label='Start of forecast')
        plt.title(title)
        plt.legend()

    def plot_one_method(self, model, title):
        """
        Plots the real data-points as well as the predictions from
        one forecasting method/model

        Parameters
        ----------
        model: str
            a string that tells the method which forecasting method/model was used
        title : str
            title of the plot
        """
        # take only the last "self.plot_size" number of points to show on plot
        train = self.train[-self.plot_size:]
        fig, ax = plt.subplots(figsize=(12, 6))
        # real data is plotted in grey
        ax.plot(self.test.index, self.test.values, color="gray")
        ax.plot(train.index, train.values, color='gray')
        # figure out which forecasting method/model was used and plot its predictions in blue
        if model == 'holt':
            ax.plot(self.test.index, self.predictions[0], label='Holts method', color='blue')
        elif model == 'arima':
            ax.plot(self.test.index, self.predictions[1], label='ARIMA model', color='blue')
        elif model == 'lstm':
            ax.plot(self.test.index, self.predictions[2], label='LSTM model', color='blue')
        # create a line that tells us when the forecasting starts
        ax.vlines(train.index[-1], 0, max(list(self.train.values) + list(self.test.values)), linestyle='--', color='r',
                  label='Start of forecast')
        plt.title(title)
        plt.legend()
