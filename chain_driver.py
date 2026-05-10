# %%writefile chain_driver.py
import mlflow
import os
import yaml
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch

# 1. RESOLVE CONFIGURATION PATH
# MLflow placed the config inside the 'code' subdirectory.
# We look for it there to match the artifact structure we just verified.
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "code", "model_config.yaml")

# Fallback: if not in /code, check the current directory (depending on MLflow version)
if not os.path.exists(config_path):
    config_path = os.path.join(current_dir, "model_config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 2. INITIALIZE COMPONENTS
chat_model = ChatDatabricks(
    endpoint=config["llm_endpoint"], 
    temperature=0.1
)

vector_store = DatabricksVectorSearch(
    index_name=config["vector_index_path"]
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. DEFINE LANGCHAIN LCEL
prompt = ChatPromptTemplate.from_messages([
    ("system", config["llm_prompt_template"]),
    ("user", "{question}")
])

model = (
    {
        "context": itemgetter("question") | retriever | format_docs, 
        "question": itemgetter("question")
    }
    | prompt
    | chat_model
    | StrOutputParser()
)

# 4. REGISTER MODEL OBJECT
mlflow.models.set_model(model)