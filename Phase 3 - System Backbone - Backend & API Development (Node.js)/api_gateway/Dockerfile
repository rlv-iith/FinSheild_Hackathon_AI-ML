# Use an official Node.js runtime
FROM node:18-slim

# Set the working directory
WORKDIR /usr/src/app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the application code
COPY . .

# The service will run on port 8080
EXPOSE 8080

# Command to run the server
CMD [ "node", "server.js" ]