FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code except files/folders listed in .dockerignore
COPY . .

# expose the port Streamlit runs on 
EXPOSE 8501
EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]