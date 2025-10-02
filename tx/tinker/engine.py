"""Background engine for processing training requests."""
import json
import time
from datetime import datetime
from sqlmodel import create_engine, Session, select

from tx.tinker.models import FutureDB, ModelDB, DB_PATH

DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, echo=False)


def process_forward_backward(request_id: str, model_id: str, request_data: dict):
    """Process a forward_backward request and return mock results."""
    # Mock implementation - returns dummy loss
    result = {
        "loss_fn_output_type": "scalar",
        "loss_fn_outputs": [{
            "loss": {
                "data": [0.5],
                "dtype": "float32",
                "shape": [1]
            }
        }],
        "metrics": {}
    }
    return result


def process_optim_step(request_id: str, model_id: str, request_data: dict):
    """Process an optim_step request and return empty result."""
    # Mock implementation - returns empty dict
    return {}


def process_pending_requests():
    """Main loop to process pending requests."""
    while True:
        try:
            with Session(engine) as session:
                # Get all pending requests
                statement = select(FutureDB).where(FutureDB.status == "pending")
                pending = session.exec(statement).all()

                for future in pending:
                    try:
                        request_data = json.loads(future.request_data)

                        # Process based on request type
                        if future.request_type == "forward_backward":
                            result_data = process_forward_backward(
                                future.request_id,
                                future.model_id,
                                request_data
                            )
                        elif future.request_type == "optim_step":
                            result_data = process_optim_step(
                                future.request_id,
                                future.model_id,
                                request_data
                            )
                        else:
                            print(f"Unknown request type: {future.request_type}")
                            continue

                        # Update the future with results
                        future.result_data = json.dumps(result_data)
                        future.status = "completed"
                        future.completed_at = datetime.utcnow()
                        session.add(future)
                        session.commit()

                        print(f"Completed {future.request_type} request {future.request_id}")

                    except Exception as e:
                        print(f"Error processing request {future.request_id}: {e}")
                        future.result_data = json.dumps({"error": str(e)})
                        future.status = "failed"
                        future.completed_at = datetime.utcnow()
                        session.add(future)
                        session.commit()

        except Exception as e:
            print(f"Error in main loop: {e}")

        # Poll every 100ms
        time.sleep(0.1)


def main():
    """Entry point for the background engine."""
    print("Starting background engine...")
    process_pending_requests()


if __name__ == "__main__":
    main()
