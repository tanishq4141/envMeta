from openenv.core.env_server import create_fastapi_app
from environment import LegalDocumentReviewEnv
from models import LegalAction, LegalObservation

app = create_fastapi_app(
    env=LegalDocumentReviewEnv,
    action_cls=LegalAction,
    observation_cls=LegalObservation,
)
