# Streamlit Deployment Guide

## Local Run
1. Open a terminal in this folder.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run app:
   - `streamlit run app.py`

## Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app.
3. Set:
   - Main file path: `app.py`
   - Python version: 3.11
4. Deploy.

## Submission Check
- Use the public URL from Streamlit Cloud (not localhost).
- Open it in incognito to confirm public access.
- Verify all four tabs load and interactive prediction works.

## Notes
- Models are pre-trained in notebook and saved under `streamlit_artifacts/`.
- The app only loads serialized artifacts and performs inference.
