from openenv.core.env_server import create_web_interface_app
from environment import LegalDocumentReviewEnv
from models import LegalAction, LegalObservation
import uvicorn

app = create_web_interface_app(
    env=LegalDocumentReviewEnv,
    action_cls=LegalAction,
    observation_cls=LegalObservation,
)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
