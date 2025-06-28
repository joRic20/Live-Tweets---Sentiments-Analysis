# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code into the container (change as needed for subfolders)
COPY . .

# By default, run bash; override CMD in docker-compose for each service
CMD ["bash"]
