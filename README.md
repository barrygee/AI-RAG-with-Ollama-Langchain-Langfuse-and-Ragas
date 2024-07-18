# GenAI Retrieval Augmented Generation (RAG) application with locally running models using Ollama, Langchain, Langfuse and Ragas

## Setup

In the root directory run 
```
docker-compose up --build
```

## Python3 Environment

Create a new Python3 virtual environment
```
python3 -m venv venv
```

Activate the new Python3 virtual environment
```
source venv/bin/activate
```

Install the required dependencies into the new Python3 virtual environment
```
pip install -r requirements.txt
```


## Langfuse

Once the Langfuse-server container is up and running;

If this is the first time you have setup the application:
- visit http://localhost:4000 in your browser
- Create a new account
- Click the 'New Project' button
- Name the project 'coreai-techspike-evaluations'
- Click 'Settings' in the left hand panel 
- Click 'Create new API keys'
- Copy the secret key and paste it into your .env file 'LANGFUSE_SECRET_KEY'
- Copy the public key and paste it into your .env file 'LANGFUSE_PUBLIC_KEY'



## PGVector

Reference: 
https://api.python.langchain.com/en/latest/vectorstores/langchain_postgres.vectorstores.PGVector.html#langchain_postgres.vectorstores.PGVector