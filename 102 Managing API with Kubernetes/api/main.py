from aiohttp import web, ClientSession
from sklearn.metrics import mean_absolute_error
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from statsmodels.tsa.statespace.sarimax import SARIMAX

import asyncio
import aiohttp_cors
import numpy as np
import model_store
import settings

app = web.Application()
routes = web.RouteTableDef()

# model = None
# model_mae = None


async def load_data():
    """Retrieves daily temperature data from KNMI."""

    # Documentation: https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script

    # Request payload (framework makes it form-encoded)
    data = {
        # Data from Eindhoven is identified with 370
        'stns': 370,
        # Only the daily average temperature (in 0.1 degrees celsius)
        'vars': 'TG',
        'start': 20100101
    }

    url = 'http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi'

    # Make the request to retrieve data from KNMI
    async with ClientSession() as session:
        async with session.post(url, data=data) as resp:
            # Response is a csv document with header information commented with '#'
            text = await resp.text()

            # For every data line, parse the columns to the data -> giving a dictionary [{date, temperature}, {...}]
            data = [{
                # First column is the station, which we ignore
                'date': line.split(',')[1].strip(),  # Date is the second column
                'temperature': int(line.split(',')[2].strip()) / 10  # Convert to normal celsius
            } for line in text.splitlines() if not line.startswith('#')]

            return data


def train_model_blocking(data):
    # Get all the temperatures
    y = [p['temperature'] for p in data]

    # Split model in train and test set
    train_len = len(y) - 30
    y_train, y_test = y[:train_len], y[train_len:]

    # Train the model on the train set
    sarimax = SARIMAX(y_train, order=(0, 0, 0), seasonal_order=(1, 1, 1, 1),
                      enforce_invertibility=False, enforce_stationarity=False)
    updated_model = sarimax.fit()

    # Make forecast with length of the test set
    y_pred = updated_model.forecast(len(y_test))

    # Calculate the MAE
    mae = mean_absolute_error(y_test, y_pred)

    # Print this value for debugging
    print('MAE: {:.2f} (avg: {:.2f} +- {:.2f})'.format(mae, np.mean(y_test), np.std(y_test)))

    # Train the model on the complete dataset to make future predictions
    sarimax = SARIMAX(y, order=(0, 0, 0), seasonal_order=(1, 1, 1, 1),
                      enforce_invertibility=False, enforce_stationarity=False)
    updated_model = sarimax.fit()
    print(updated_model.summary())

    return updated_model, mae


@routes.get('/data')
async def get_data(request):
    """Endpoint which retrieves temperature data and returns it in raw form to the client."""

    # async tells python to continue when the function is complete, in the meantime however it is
    # possible to serve other requests!
    data = await load_data()

    # data is an array with temperature date, return it as json to the client
    return web.json_response(data)


@routes.post('/train-model')
async def post_train_model(request):
    # Retrieve the data from KNMI
    data = await load_data()

    # Because training the model is a blocking operation, we delegate it to a different thread. This
    # allows the current program to continue (e.g. handle other requests) without any interuptions.
    with ProcessPoolExecutor() as pool:
        model, mae = await asyncio.get_event_loop().run_in_executor(pool, partial(train_model_blocking, data))

    # Save the model in the database
    await model_store.save(model, mae)

    return web.Response(text='Model updated with MAE of {:.2f}'.format(mae))


@routes.get('/predictions')
async def get_predictions(request):
    model = await model_store.get_latest()

    if model is None:
        return web.HTTPBadRequest(reason='Model is not trained yet!')

    days = int(request.query.get('days', 7))
    if days <= 0:
        return web.HTTPBadRequest(reason='Parameter days must be at least 1.')

    temperatures = list(model.forecast(days))
    return web.json_response(temperatures)


app.add_routes(routes)

# Configure default CORS settings.
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    )
})

# Configure CORS on all routes.
for route in app.router.routes():
    cors.add(route)


web.run_app(app, port=7001)
