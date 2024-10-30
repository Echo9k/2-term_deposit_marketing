import numpy as np
import plotly.express as px


def plot_true_vs_predicted(y_test, y_predict, title="True vs Predicted"):
    """
    Plots the relationship between true and predicted values, showing prediction error,
    point sizes based on the frequency of each (True, Predicted) pair, and includes
    the count of values at each point in the hover information.
    
    Parameters:
    - y_test: array-like, the true values.
    - y_predict: array-like, the predicted values.
    - title: str, optional, the title of the plot.
    """
    # Calculate the error between true and predicted values
    errors = np.abs(y_test - y_predict)

    # Calculate the frequency of each (True, Predicted) pair
    unique_points, counts = np.unique(np.column_stack((y_test, y_predict)), axis=0, return_counts=True)
    
    # Create a dictionary mapping each unique (True, Predicted) pair to its frequency
    size_dict = {tuple(point): np.power(count, 1/8) for point, count in zip(unique_points, counts)}
    
    # Determine point sizes based on the frequency of each (True, Predicted) pair
    point_sizes = np.array([size_dict[(true, pred)] for true, pred in zip(y_test, y_predict)])
    
    # Create the scatter plot
    fig = px.scatter(
        x=y_test, 
        y=y_predict, 
        labels={'x': 'True', 'y': 'Predicted'}, 
        title=title
    )

    # Add hover information with the count of values at each point
    hover_text = [f'True: {true}<br>Predicted: {pred}<br>Error: {error:.2f}<br>Count: {size_dict[(true, pred)]**8:.0f}' 
                  for true, pred, error in zip(y_test, y_predict, errors)]

    # Update the trace with point sizes, hover information, and color scaling based on errors
    fig.update_traces(
        marker=dict(
            size=point_sizes * 10,  # Adjust size scaling
            opacity=0.7, 
            color=errors,  # Color by prediction error
            colorscale='Viridis', 
            showscale=True,
            colorbar=dict(title="Prediction Error")
        ),
        hovertemplate=hover_text
    )

    # Add a diagonal line for reference (perfect prediction line)
    fig.add_shape(
        type="line",
        x0=min(y_test), x1=max(y_test), 
        y0=min(y_test), y1=max(y_test),
        line=dict(color='red', dash='dash')
    )

    # Automatically limit the axes range based on the max value between y_test and y_predict
    max_value = max(max(y_test), max(y_predict))
    fig.update_layout(xaxis_range=[0, max_value], yaxis_range=[0, max_value])

    # Show the plot
    fig.show()



# def train_model(X_train, y_train, metric="mae", **kwargs):
#     """
#     Trains an AutoML model on the given data.
#     """
#     # Train with labeled input data
#     automl.fit(X_train=X_train, y_train=y_train, metric=metric, **kwargs)

#     # Print the best model
#     print(automl.model.estimator)

#     # Save the model    
#     # automl.model.save("../models/flaml/model.pkl")

#     # Assume you have trained a model and are testing it
#     # automl.predict(X_test)

#     return automl

# def train_model(X_train, y_train, metric="r2", **kwargs): 
#     """
#     Trains an AutoML model on the given data and logs the run in MLflow.
#     """
#     automl = AutoML()  # Initialize the AutoML object within the task
#     automl.fit(X_train=X_train, y_train=y_train, metric=metric, **kwargs)

#     # Return relevant model info without logging here to avoid MLflow-related issues in Ray
#     return {
#         "metric": metric,
#         "best_estimator": automl.best_estimator,
#         "best_config": automl.best_config,
#         "best_loss": automl.best_loss
#     }
