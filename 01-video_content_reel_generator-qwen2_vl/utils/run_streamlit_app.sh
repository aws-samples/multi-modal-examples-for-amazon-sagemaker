#!/bin/bash

sudo apt-get install -y jq
# Run the Streamlit app and push it to the background
streamlit run streamlit-ui/app.py &

DOMAIN_ID=$(jq -r '.DomainId' /opt/ml/metadata/resource-metadata.json)
SPACE_NAME=$(jq -r '.SpaceName' /opt/ml/metadata/resource-metadata.json)
STREAMLIT_URL=$(aws sagemaker describe-space --domain-id $DOMAIN_ID --space-name $SPACE_NAME | jq -r '.Url')

echo "=====>  Launch Streamlit: $STREAMLIT_URL/proxy/8501/"

# Wait for Streamlit to initialize
sleep 5

# Run the Python script
python3 docker-artifacts/inference.py