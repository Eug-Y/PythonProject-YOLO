"""Launch the deployed farm-analysis web service on port 5055."""

from waitress import serve

from web_viewer import app


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5055, threads=4, channel_timeout=600)
