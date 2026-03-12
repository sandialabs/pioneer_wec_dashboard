# Pioneer WEC dashboard
A simple dashboard for viewing up-to-date data from the Pioneer WEC v1 prototype.

## Usage
Build the dashboard site locally with the steps below.

### Prerequisites
- Python 3.10+ (recommended)
- Ruby + Bundler (for Jekyll)

### 1) Install Python dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Install Ruby/Jekyll dependencies
```bash
bundle install
```

### 3) Build the site
Run the full pipeline (fetch data, generate plots, and build the Jekyll site):

```bash
python app.py all
```

### 4) View build output
The generated site files are written to:

- `output/`

Open `output/index.html` in a browser to inspect the locally built site.

## Build options
Partial builds can be run by passing optional arguments when running `app.py`:

- `python app.py` or `python app.py all`: run the full pipeline (fetch data, generate plots, build site)
- `python app.py fetch-data --start-date YYYY-MM-DD`: fetch and cache raw data starting from a date (default: 2025-11-03)
- `python app.py generate-plots`: generate plot HTML files from cached data
- `python app.py build-site`: generate Jekyll includes and build/copy the site output

## Data caching
To alleviate the need to pull data during debugging and to speed up GitHub actions, raw data is cached locally in a `.cache` directory.
Additionally, processed data files are saved in `output/data/` to speed up partial builds and allow for the user to perform custom analyses.
