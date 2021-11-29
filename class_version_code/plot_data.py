import matplotlib.pyplot as plt
import pandas as pd


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


def main():
    data = pd.read_csv('..\\prepared_data\\DBS_SYSCO_10Y_Prepared.csv', header=0)
    filtered_data = data[data["MD Material Group"] == "CHBOC"].values[0][1:]
    new_index = pd.to_datetime(data.columns[1:])
    unit_data = pd.Series(filtered_data, new_index)

    plt.figure(figsize=(16, 10), dpi=80)
    plt.plot(unit_data.index[20:], unit_data.values[20:], color="tab:blue")
    #plt.ylim(50, 750)
    xtick_location = unit_data.index.tolist()[::12]
    #xtick_labels = [x[-4:] for x in unit_data.index.tolist()[::12]]
    plt.xticks(rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title("CHBOC number of units (2012-2021)", fontsize=22)
    plt.grid(axis='both', alpha=.3)

    plt.gca().spines["top"].set_alpha(0.0)
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)
    plt.gca().spines["left"].set_alpha(0.3)
    plt.show()


if __name__ == "__main__":
    main()
