# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
    
# Copy the requirements file into the container at /app
COPY west/requirements.txt .



# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model into the container
COPY . .

# Expose the port the Flask app runs on
EXPOSE 5000

# Set environment variables (optional, but good practice)
ENV FLASK_APP=west.app
ENV FLASK_ENV=production

# Define the command to run the Flask application
CMD ["python", "-m", "west.app"]
