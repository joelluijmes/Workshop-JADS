from bson import ObjectId
from logging import getLogger
from datetime import datetime
from collections import namedtuple

import pickle
import motor.motor_asyncio
import settings

logger = getLogger('data.models')

Model = namedtuple('Model', ['createdAt', 'mae', 'model'])


def _get_db():
    """Returns the database connection."""

    # Create client to mongodb server
    client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB)
    print(settings.MONGODB)

    # Return connection to the database
    return client.weather


async def save(model, mae):
    doc = {
        'createdAt': datetime.utcnow(),
        'mae': mae,
        'model': pickle.dumps(model)
    }

    await _get_db().models.insert_one(doc)


async def get_latest():
    doc = await _get_db().models.find_one(sort=[('createdAt', -1)])
    return pickle.loads(doc['model']) if doc else None
