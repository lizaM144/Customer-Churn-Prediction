# Base Image: Use a lightweight Python version
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (for caching)
COPY requirements.txt .

# install the dependencies
# --no-cache-dir keeps the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code except files/folders listed in .dockerignore
COPY . .

# expose the port Streamlit runs on (default is 8501)
EXPOSE 8501
EXPOSE 8000

# The command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]