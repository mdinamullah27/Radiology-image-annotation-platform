# Radiology Image Annotation Dashboard

## Overview

The Radiology Image Annotation Dashboard is a web-based application built with Dash and Python, designed for medical students and radiologists to practice annotating medical images (DICOM, PNG, JPG, JPEG). Users can upload images, draw annotations (polygons, rectangles, circles, lines, freehand), add text labels, analyze regions with AI (via OpenAI API), and export annotations or generate reports. The app supports image transformations (zoom, rotate, flip), 3D visualization, and a library for saving and loading images.

Key features:

- Upload and process DICOM or standard image files (resized to 512x512 pixels).
- Interactive annotation tools with statistics (area, coverage, bounding box).
- AI-powered analysis of full images or annotated regions using OpenAI's API.
- Save images to a library with thumbnails.
- Export annotations as ZIP files or generate HTML reports.
- Cache clearing on browser reload to reset state.

## Prerequisites

- **Python 3.8+**
- **Dependencies**: Install required Python packages listed in `requirements.txt`.
- **OpenAI API Key**: Required for AI analysis features (set as environment variable `OPENAI_API_KEY`).
- **Fonts**: Ensure `arial.ttf`, `times.ttf`, and `cour.ttf` are available in the system font directory for text annotations (or adjust `add_text_to_image` to use available fonts).
- **Browser**: Modern browser (Chrome, Firefox, Edge) for best compatibility.

## Installation

1. **Clone the Repository** (or download the project files):

   ```bash
   git clone <repository-url>
   cd radiology-annotation-dashboard
   ```

2. **Create a Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**: Create a `requirements.txt` file with the following content:

   ```
   anyio==4.11.0
   blinker==1.9.0
   certifi==2025.8.3
   charset-normalizer==3.4.3
   click==8.3.0
   dash==3.2.0
   dash-bootstrap-components==2.0.4
   dash_canvas==0.1.0
   dotenv==0.9.9
   Flask==3.1.2
   gunicorn==23.0.0
   h11==0.16.0
   httptools==0.6.4
   idna==3.10
   imageio==2.37.0
   importlib_metadata==8.7.0
   itsdangerous==2.2.0
   Jinja2==3.1.6
   joblib==1.5.2
   lazy_loader==0.4
   MarkupSafe==3.0.2
   narwhals==2.5.0
   nest-asyncio==1.6.0
   networkx==3.5
   numpy==2.3.3
   packaging==25.0
   pillow==11.3.0
   plotly==6.3.0
   pydicom==3.0.1
   python-dotenv==1.1.1
   PyYAML==6.0.3
   requests==2.32.5
   retrying==1.4.2
   scikit-image==0.25.2
   scikit-learn==1.7.2
   scipy==1.16.2
   setuptools==80.9.0
   sniffio==1.3.1
   threadpoolctl==3.6.0
   tifffile==2025.9.20
   typing_extensions==4.15.0
   urllib3==2.5.0
   uvicorn==0.37.0
   uvloop==0.21.0
   watchfiles==1.1.0
   websockets==15.0.1
   Werkzeug==3.1.3
   zipp==3.23.0
   ```

   Then install:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**: Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Replace `your_openai_api_key_here` with your actual OpenAI API key.

5. **Directory Setup**: The app automatically creates the following directories:

   - `sessions/`: Stores session data as pickle files.
   - `case/`: Stores saved images.
   - `case/thumbnails/`: Stores image thumbnails.

   Ensure write permissions for these directories.

## Running the Application

1. **Start the Server**: Run the main script (e.g., `app.py`):

   ```bash
   python app.py
   ```

   The app will start on `http://192.168.1.144:8050` (as configured). To change the host or port, modify the `app.run` line:

   ```python
   app.run(debug=True, host='localhost', port=8050)
   ```

2. **Access the App**: Open your browser and navigate to `http://192.168.1.144:8050` (or the configured host/port).

## Usage

1. **Uploading Images**:

   - Click "Upload Image" to upload a DICOM (`.dcm`), PNG, JPG, or JPEG file.
   - Uploaded images are automatically resized to 512x512 pixels.
   - Image details (filename, type, size) are displayed below the canvas.

2. **Annotating Images**:

   - Use the toolbar to select tools: Rectangle, Circle, Line, Freehand, or Labeling (text).
   - Draw annotations on the canvas. Statistics (tool type, area, coverage, bounding box) appear in the "Stats" panel.
   - For text annotations, a modal prompts for text, font size, style, and position.
   - Use the Eraser tool to remove parts of annotations.

3. **Image Transformations**:

   - Zoom in/out or fit to screen using the zoom buttons.
   - Rotate (left/right) or flip (horizontal/vertical) the image.
   - View a 3D surface plot of image intensity via the "3D View" button.

4. **Saving and Loading**:

   - Click "Save to Library" to store the image in the `case/` directory with a thumbnail.
   - Click "Library" to view and load saved images via thumbnails.
   - Save annotations to the session state with the "Save" button.

5. **AI Analysis**:

   - Click "Analyze" to get AI analysis of the full image or selected region (requires OpenAI API key).
   - Results appear in the chat box, with related Radiopaedia cases in the "Analyze" panel.
   - Enter custom prompts in the chat input for specific analysis.

6. **Exporting**:

   - Click "Export ZIP" to download a ZIP file containing the annotated image, mask, and metadata.
   - Click "Generate Report" to download an HTML report with annotation details.

7. **Cache Clearing**:

   - The app clears its cache and resets state on every browser reload, ensuring no previous data persists.

## Project Structure

```
radiology-annotation-dashboard/
├── app.py           # Main application script
├── .env                  # Environment variables (create with OPENAI_API_KEY)
├── requirements.txt      # Python dependencies
├── sessions/             # Session data (auto-created)
├── case/                 # Saved images (auto-created)
│   └── thumbnails/       # Image thumbnails (auto-created)
```

## Notes

- **Cache Clearing**: The app resets the global `state` dictionary and clears browser `localStorage`/`sessionStorage` on reload to prevent data persistence. Old session files are deleted on server startup.
- **DICOM Support**: Only single-frame DICOMs are supported. Multi-frame DICOMs take the first slice.
- **Fonts**: If `arial.ttf`, `times.ttf`, or `cour.ttf` are unavailable, the app falls back to default fonts. Install these fonts or modify `add_text_to_image` for alternatives.
- **Multi-User Limitation**: The global `state` dictionary is shared across users. For multi-user deployment, refactor to use `dcc.Store` with session storage.
- **HTTPS**: The `caches` API (used in cache clearing) requires HTTPS. For local development, it’s skipped if unavailable.
- **Debug Mode**: Running with `debug=True` enables hot-reloading, which may reset state unexpectedly. Use `debug=False` for production.

## Troubleshooting

- **JavaScript Errors**: Check the browser console (F12 &gt; Console) for errors. Ensure `caches` API issues are handled (see client-side callback).
- **Image Upload Fails**: Verify file format (DICOM, PNG, JPG, JPEG) and check console for PIL/pydicom errors.
- **AI Analysis Fails**: Ensure `OPENAI_API_KEY` is set correctly in `.env`. Check network connectivity for API requests.
- **Cache Not Clearing**: Confirm `dcc.Location`, `reset-trigger`, and `client-reset-trigger` are in the layout. Verify client-side callback logs.
- **Font Errors**: If text annotations fail, ensure specified fonts are available or update `add_text_to_image` to use system fonts.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## Contact

For questions or issues, please open an issue on the repository or contact the maintainer.# Radiology Image Annotation Dashboard

## Overview

The Radiology Image Annotation Dashboard is a web-based application built with Dash and Python, designed for medical students and radiologists to practice annotating medical images (DICOM, PNG, JPG, JPEG). Users can upload images, draw annotations (polygons, rectangles, circles, lines, freehand), add text labels, analyze regions with AI (via OpenAI API), and export annotations or generate reports. The app supports image transformations (zoom, rotate, flip), 3D visualization, and a library for saving and loading images.

Key features:

- Upload and process DICOM or standard image files (resized to 512x512 pixels).
- Interactive annotation tools with statistics (area, coverage, bounding box).
- AI-powered analysis of full images or annotated regions using OpenAI's API.
- Save images to a library with thumbnails.
- Export annotations as ZIP files or generate HTML reports.
- Cache clearing on browser reload to reset state.

## Prerequisites

- **Python 3.8+**
- **Dependencies**: Install required Python packages listed in `requirements.txt`.
- **OpenAI API Key**: Required for AI analysis features (set as environment variable `OPENAI_API_KEY`).
- **Fonts**: Ensure `arial.ttf`, `times.ttf`, and `cour.ttf` are available in the system font directory for text annotations (or adjust `add_text_to_image` to use available fonts).
- **Browser**: Modern browser (Chrome, Firefox, Edge) for best compatibility.

## Installation

1. **Clone the Repository** (or download the project files):

   ```bash
   git clone <repository-url>
   cd radiology-annotation-dashboard
   ```

2. **Create a Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**: Create a `requirements.txt` file with the following content:

   ```
   anyio==4.11.0
   blinker==1.9.0
   certifi==2025.8.3
   charset-normalizer==3.4.3
   click==8.3.0
   dash==3.2.0
   dash-bootstrap-components==2.0.4
   dash_canvas==0.1.0
   dotenv==0.9.9
   Flask==3.1.2
   gunicorn==23.0.0
   h11==0.16.0
   httptools==0.6.4
   idna==3.10
   imageio==2.37.0
   importlib_metadata==8.7.0
   itsdangerous==2.2.0
   Jinja2==3.1.6
   joblib==1.5.2
   lazy_loader==0.4
   MarkupSafe==3.0.2
   narwhals==2.5.0
   nest-asyncio==1.6.0
   networkx==3.5
   numpy==2.3.3
   packaging==25.0
   pillow==11.3.0
   plotly==6.3.0
   pydicom==3.0.1
   python-dotenv==1.1.1
   PyYAML==6.0.3
   requests==2.32.5
   retrying==1.4.2
   scikit-image==0.25.2
   scikit-learn==1.7.2
   scipy==1.16.2
   setuptools==80.9.0
   sniffio==1.3.1
   threadpoolctl==3.6.0
   tifffile==2025.9.20
   typing_extensions==4.15.0
   urllib3==2.5.0
   uvicorn==0.37.0
   uvloop==0.21.0
   watchfiles==1.1.0
   websockets==15.0.1
   Werkzeug==3.1.3
   zipp==3.23.0
   ```

   Then install:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**: Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Replace `your_openai_api_key_here` with your actual OpenAI API key.

5. **Directory Setup**: The app automatically creates the following directories:

   - `sessions/`: Stores session data as pickle files.
   - `case/`: Stores saved images.
   - `case/thumbnails/`: Stores image thumbnails.

   Ensure write permissions for these directories.

## Running the Application

1. **Start the Server**: Run the main script (e.g., `app.py`):

   ```bash
   python app.py
   ```

   The app will start on `http://192.168.1.144:8050` (as configured). To change the host or port, modify the `app.run` line:

   ```python
   app.run(debug=True, host='localhost', port=8050)
   ```

2. **Access the App**: Open your browser and navigate to `http://192.168.1.144:8050` (or the configured host/port).

## Usage

1. **Uploading Images**:

   - Click "Upload Image" to upload a DICOM (`.dcm`), PNG, JPG, or JPEG file.
   - Uploaded images are automatically resized to 512x512 pixels.
   - Image details (filename, type, size) are displayed below the canvas.

2. **Annotating Images**:

   - Use the toolbar to select tools: Rectangle, Circle, Line, Freehand, or Labeling (text).
   - Draw annotations on the canvas. Statistics (tool type, area, coverage, bounding box) appear in the "Stats" panel.
   - For text annotations, a modal prompts for text, font size, style, and position.
   - Use the Eraser tool to remove parts of annotations.

3. **Image Transformations**:

   - Zoom in/out or fit to screen using the zoom buttons.
   - Rotate (left/right) or flip (horizontal/vertical) the image.
   - View a 3D surface plot of image intensity via the "3D View" button.

4. **Saving and Loading**:

   - Click "Save to Library" to store the image in the `case/` directory with a thumbnail.
   - Click "Library" to view and load saved images via thumbnails.
   - Save annotations to the session state with the "Save" button.

5. **AI Analysis**:

   - Click "Analyze" to get AI analysis of the full image or selected region (requires OpenAI API key).
   - Results appear in the chat box, with related Radiopaedia cases in the "Analyze" panel.
   - Enter custom prompts in the chat input for specific analysis.

6. **Exporting**:

   - Click "Export ZIP" to download a ZIP file containing the annotated image, mask, and metadata.
   - Click "Generate Report" to download an HTML report with annotation details.

7. **Cache Clearing**:

   - The app clears its cache and resets state on every browser reload, ensuring no previous data persists.

## Project Structure

```
radiology-annotation-dashboard/
├── app.py           # Main application script
├── .env                  # Environment variables (create with OPENAI_API_KEY)
├── requirements.txt      # Python dependencies
├── sessions/             # Session data (auto-created)
├── case/                 # Saved images (auto-created)
│   └── thumbnails/       # Image thumbnails (auto-created)
```

## Notes

- **Cache Clearing**: The app resets the global `state` dictionary and clears browser `localStorage`/`sessionStorage` on reload to prevent data persistence. Old session files are deleted on server startup.
- **DICOM Support**: Only single-frame DICOMs are supported. Multi-frame DICOMs take the first slice.
- **Fonts**: If `arial.ttf`, `times.ttf`, or `cour.ttf` are unavailable, the app falls back to default fonts. Install these fonts or modify `add_text_to_image` for alternatives.
- **Multi-User Limitation**: The global `state` dictionary is shared across users. For multi-user deployment, refactor to use `dcc.Store` with session storage.
- **HTTPS**: The `caches` API (used in cache clearing) requires HTTPS. For local development, it’s skipped if unavailable.
- **Debug Mode**: Running with `debug=True` enables hot-reloading, which may reset state unexpectedly. Use `debug=False` for production.

## Troubleshooting

- **JavaScript Errors**: Check the browser console (F12 &gt; Console) for errors. Ensure `caches` API issues are handled (see client-side callback).
- **Image Upload Fails**: Verify file format (DICOM, PNG, JPG, JPEG) and check console for PIL/pydicom errors.
- **AI Analysis Fails**: Ensure `OPENAI_API_KEY` is set correctly in `.env`. Check network connectivity for API requests.
- **Cache Not Clearing**: Confirm `dcc.Location`, `reset-trigger`, and `client-reset-trigger` are in the layout. Verify client-side callback logs.
- **Font Errors**: If text annotations fail, ensure specified fonts are available or update `add_text_to_image` to use system fonts.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## Contact

For questions or issues, please open an issue on the repository or contact the maintainer.
