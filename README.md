# Excel Data Query Interface

This project is a web application that allows users to upload Excel files and query their contents using natural language. It leverages a Flask backend for processing and a Streamlit frontend for user interaction.

## Features

- Upload Excel files in various formats (.xlsx, .xls, .xlsb).
- Clean and transform data using pandas.
- Store data in a SQLite database.
- Query data using natural language, translated to SQL via NLP models.
- View results in a user-friendly interface.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/David-Lee23/Data-Handling-With-BasicAI
   ```
2. **Navigate to the project directory**:
   ```bash
   cd /Data-Handling-With-BasicAI
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the backend server**:
   ```bash
   python app.py
   ```
2. **Run the frontend**:
   ```bash
   streamlit run frontend.py
   ```
3. **Access the application**:
   Open your web browser and go to `http://localhost:8501` to use the application.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.
