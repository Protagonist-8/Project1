# Use an official Python runtime
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port used by Streamlit
EXPOSE 7860

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
